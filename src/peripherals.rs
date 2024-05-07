// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Generic, memory-mapped peripherals implemented using runtime callbacks.
use crate::configuration::Callback;
use crate::Cpu;
use ndarray::{s, Array1, Array2, Array3};
use std::{
    cell::Cell,
    convert::{TryFrom, TryInto},
    sync::atomic::{AtomicI32, AtomicU32, Ordering},
};
use PeriphReq::{Load, Store};

/// Reference held by execution engine, referencing each peripheral instance in each cluster
pub struct Peripherals {
    peripherals: Vec<Box<dyn Peripheral>>,
    cluster_peripherals: Vec<Vec<(u32, usize)>>,
}

unsafe impl Sync for Peripherals {}

impl Peripherals {
    pub fn new() -> Self {
        Self {
            peripherals: get_peripheral_types(),
            cluster_peripherals: Default::default(),
        }
    }

    pub fn add_cluster(&mut self, callbacks: &Vec<Callback>) {
        self.cluster_peripherals.push(
            callbacks
                .iter()
                .map(|x| {
                    (
                        x.size,
                        self.peripherals
                            .iter()
                            .position(|p| x.name.eq(&p.get_name()))
                            .expect(&format!("Undefined peripheral type: {}", x.name)[..]),
                    )
                })
                .collect(),
        );
    }

    pub fn load(&self, cpu: &Cpu, cluster_id: usize, addr: u32, size: u8) -> u32 {
        self.load_store(cpu, cluster_id, addr, size, Load)
    }

    pub fn store(&self, cpu: &Cpu, cluster_id: usize, addr: u32, value: u32, mask: u32, size: u8) {
        self.load_store(cpu, cluster_id, addr, size, Store(value, mask));
    }

    fn load_store(
        &self,
        cpu: &Cpu,
        cluster_id: usize,
        mut addr: u32,
        size: u8,
        req: PeriphReq,
    ) -> u32 {
        for i in &self.cluster_peripherals[cluster_id] {
            if addr < i.0 {
                return match req {
                    Load => {
                        trace!(
                            "Periph load from {}: cluster_id {}, offs 0x{:x}, size {}",
                            self.peripherals[i.1].get_name(),
                            cluster_id,
                            addr,
                            size
                        );
                        self.peripherals[i.1].load(cpu, addr, size)
                    }
                    Store(val, mask) => {
                        trace!(
                            "Periph store to {}: cluster_id {}, offs 0x{:x}, size {}, mask 0x{:x}, val {}",
                            self.peripherals[i.1].get_name(),
                            cluster_id,
                            addr,
                            size,
                            mask,
                            val
                        );
                        self.peripherals[i.1].store(cpu, addr, val, mask, size);
                        0
                    }
                };
            }
            addr = addr - i.0;
        }
        // Handle unmapped accesses: have no side effect on peripherals
        // TODO: should we trigger an error-response-like exception here?
        match req {
            Load => trace!(
                "Unmapped periph load: cluster_id {}, addr {}, size {}",
                cluster_id,
                addr,
                size
            ),
            Store(val, mask) => trace!(
                "Unmapped periph store: cluster_id {}, addr {}, size {}, mask {}, val {}",
                cluster_id,
                addr,
                size,
                mask,
                val
            ),
        }
        0
    }
}

enum PeriphReq {
    Load,
    Store(u32, u32),
}

/// Trait representing a peripheral
pub trait Peripheral {
    /// should return the same name as in the config file
    fn get_name(&self) -> &'static str;
    /// store instruction
    fn store(&self, cpu: &Cpu, addr: u32, value: u32, mask: u32, size: u8);
    /// load instruction
    fn load(&self, cpu: &Cpu, addr: u32, size: u8) -> u32;
}

/// Function called by the cpu to get the peripheral types. This function should
/// return a vector containing an instance of each available peripherable type.
/// To add a new peripheral type, declare it below and add it here.
pub fn get_peripheral_types() -> Vec<Box<dyn Peripheral>> {
    vec![
        Box::new(Semaphores::default()),
        Box::new(Fence::default()),
        Box::new(ZeroMemory::default()),
        Box::new(MemPoolDMA::default()),
        Box::new(MemPoolITA::default()),
        Box::new(HWPEITA::default()),
    ]
}

#[derive(Default)]
struct Fence {
    set: AtomicU32,
    current: AtomicU32,
}

impl Peripheral for Fence {
    fn get_name(&self) -> &'static str {
        "fence"
    }

    fn store(&self, _cpu: &Cpu, addr: u32, val: u32, _mask: u32, _: u8) {
        match addr {
            0x0 => self.set.store(val, Ordering::SeqCst),
            _ => self.current.store(val, Ordering::SeqCst),
        }
    }

    fn load(&self, _cpu: &Cpu, _: u32, _: u8) -> u32 {
        self.current.fetch_add(1, Ordering::SeqCst);
        while self.set.load(Ordering::SeqCst) != self.current.load(Ordering::SeqCst) {}
        0
    }
}

#[derive(Default)]
struct Semaphores {
    empty_count: AtomicU32,
    full_count: AtomicU32,
    use_queue: AtomicU32,
}

impl Peripheral for Semaphores {
    fn get_name(&self) -> &'static str {
        "semaphores"
    }

    fn store(&self, _cpu: &Cpu, addr: u32, val: u32, _mask: u32, _: u8) {
        match addr {
            0x0 => self.empty_count.store(val, Ordering::SeqCst),
            0x4 => {
                self.empty_count.fetch_add(val, Ordering::SeqCst);
            }
            0x8 => {
                while self
                    .empty_count
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                        if x >= val {
                            Some(x - val)
                        } else {
                            None
                        }
                    })
                    .is_err()
                {}
            }
            0xc => self.full_count.store(val, Ordering::SeqCst),
            0x10 => {
                self.full_count.fetch_add(val, Ordering::SeqCst);
            }
            0x14 => {
                while self
                    .full_count
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                        if x >= val {
                            Some(x - val)
                        } else {
                            None
                        }
                    })
                    .is_err()
                {}
            }
            0x18 => self.use_queue.store(val, Ordering::SeqCst),
            0x1c => {
                self.use_queue.fetch_add(val, Ordering::SeqCst);
            }
            _ => {
                while self
                    .use_queue
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                        if x >= val {
                            Some(x - val)
                        } else {
                            None
                        }
                    })
                    .is_err()
                {}
            }
        }
    }

    fn load(&self, _cpu: &Cpu, _: u32, _: u8) -> u32 {
        0
    }
}

#[derive(Default)]
struct ZeroMemory {}

impl Peripheral for ZeroMemory {
    fn get_name(&self) -> &'static str {
        "zero-memory"
    }

    fn store(&self, _cpu: &Cpu, _: u32, _: u32, _: u32, _: u8) {}

    fn load(&self, _cpu: &Cpu, _: u32, _: u8) -> u32 {
        0
    }
}

#[derive(Default)]
struct MemPoolDMA {
    src_addr: AtomicU32,
    dst_addr: AtomicU32,
    num_bytes: AtomicU32,
    conf: AtomicU32,
    status: AtomicU32,
    next_id: AtomicU32,
    done: AtomicU32,
}

impl Peripheral for MemPoolDMA {
    /// should return the same name as in the config file
    fn get_name(&self) -> &'static str {
        "mempool-dma"
    }
    /// store instruction
    fn store(&self, _cpu: &Cpu, addr: u32, value: u32, _mask: u32, _size: u8) {
        match addr {
            0x00 => self.src_addr.store(value, Ordering::SeqCst),
            0x04 => self.dst_addr.store(value, Ordering::SeqCst),
            0x08 => self.num_bytes.store(value, Ordering::SeqCst),
            0x0C => self.conf.store(value, Ordering::SeqCst),
            0x10 => (), /* status: Write has no effect */
            0x14 => (), /* next_id: Write has no effect */
            0x18 => (), /* done: Write has no effect */
            _ => unimplemented!(),
        }
        self.done.store(0, Ordering::SeqCst);
    }
    /// load instruction
    fn load(&self, cpu: &Cpu, addr: u32, _size: u8) -> u32 {
        match addr {
            0x00 => self.src_addr.load(Ordering::SeqCst),
            0x04 => self.dst_addr.load(Ordering::SeqCst),
            0x08 => self.num_bytes.load(Ordering::SeqCst),
            0x0C => self.conf.load(Ordering::SeqCst),
            0x10 => self.status.load(Ordering::SeqCst),
            0x14 => {
                cpu.binary_memcpy(
                    self.dst_addr.load(Ordering::SeqCst),
                    self.src_addr.load(Ordering::SeqCst),
                    self.num_bytes.load(Ordering::SeqCst),
                );
                self.done.store(1, Ordering::SeqCst);
                self.next_id.load(Ordering::SeqCst)
            }
            0x18 => self.done.load(Ordering::SeqCst),
            _ => unimplemented!(),
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
enum ITAStep {
    Q,
    K,
    V,
    QK,
    AV,
    OW,
    IDLE,
}

impl TryFrom<u32> for ITAStep {
    type Error = ();

    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            x if x == ITAStep::Q as u32 => Ok(ITAStep::Q),
            x if x == ITAStep::K as u32 => Ok(ITAStep::K),
            x if x == ITAStep::V as u32 => Ok(ITAStep::V),
            x if x == ITAStep::QK as u32 => Ok(ITAStep::QK),
            x if x == ITAStep::AV as u32 => Ok(ITAStep::AV),
            x if x == ITAStep::OW as u32 => Ok(ITAStep::OW),
            x if x == ITAStep::IDLE as u32 => Ok(ITAStep::IDLE),
            _ => Err(()),
        }
    }
}

struct HWPEITA {
    // HWPE ITA Registers
    trigger: AtomicU32,      // Offset 0x00
    acquire: AtomicU32,      // Offset 0x04
    evt_enable: AtomicU32,   // Offset 0x08
    status: AtomicU32,       // Offset 0x0C
    running_jobs: AtomicU32, // Offset 0x10
    soft_clear: AtomicU32,   // Offset 0x14

    input_ptr: AtomicU32,   // Offset 0x20
    weight_ptr0: AtomicU32, // Offset 0x24
    weight_ptr1: AtomicU32, // Offset 0x28
    bias_ptr: AtomicU32,    // Offset 0x2C
    output_ptr: AtomicU32,  // Offset 0x30
    seq_len: AtomicU32,     // Offset 0x34
    ita_tiles: AtomicU32,   // Offset 0x38 // tile_s [3:0], tile_e [7:4], tile_p [11:8]
    eps_mult1: AtomicU32, // Offset 0x3C // eps_mult[0] [7:0], eps_mult[1] [15:8], eps_mult[2] [23:16], eps_mult[3] [31:24]
    eps_mult2: AtomicU32, // Offset 0x40 // eps_mult[4] [7:0], eps_mult[5] [15:8]
    eps_shift1: AtomicU32, // Offset 0x44 // right_shift[0] [7:0], right_shift[1] [15:8], right_shift[2] [23:16], right_shift[3] [31:24]
    eps_shift2: AtomicU32, // Offset 0x48 // right_shift[4] [7:0], right_shift[5] [15:8]
    eps_add1: AtomicI32, // Offset 0x4C // add[0] [7:0], add[1] [15:8], add[2] [23:16], add[3] [31:24]
    eps_add2: AtomicI32, // Offset 0x50 // add[4] [7:0], add[5] [15:8]
    ctrl_stream: AtomicU32, // Offset 0x54 // ctrl_stream [0]: weight preload, ctrl_stream [1]: weight nextload, ctrl_stream [2]: bias disable, ctrl_stream [3]: bias direction

    // Internal buffers
    _input_buffer: Cell<Array2<i8>>,
    _weight_buffer0: Cell<Array2<i8>>,
    _weight_buffer1: Cell<Array2<i8>>,
    _bias_buffer: Cell<Array1<i32>>,
    _output_buffer_i32: Cell<Array2<i32>>,
    _output_buffer_i32_prev: Cell<Array2<i32>>,
    _output_buffer_i8: Cell<Array2<i8>>,

    // Internal state
    _step: AtomicU32,
    _inner_tile: AtomicU32,
}

impl Default for HWPEITA {
    fn default() -> Self {
        Self {
            // HWPE ITA Registers
            trigger: AtomicU32::new(0),
            acquire: AtomicU32::new(0),
            evt_enable: AtomicU32::new(0),
            status: AtomicU32::new(0),
            running_jobs: AtomicU32::new(0),
            soft_clear: AtomicU32::new(0),
            input_ptr: AtomicU32::new(0),
            weight_ptr0: AtomicU32::new(0),
            weight_ptr1: AtomicU32::new(0),
            bias_ptr: AtomicU32::new(0),
            output_ptr: AtomicU32::new(0),
            seq_len: AtomicU32::new(0),
            ita_tiles: AtomicU32::new(0),
            eps_mult1: AtomicU32::new(0),
            eps_mult2: AtomicU32::new(0),
            eps_shift1: AtomicU32::new(0),
            eps_shift2: AtomicU32::new(0),
            eps_add1: AtomicI32::new(0),
            eps_add2: AtomicI32::new(0),
            ctrl_stream: AtomicU32::new(0),

            // Internal buffers
            _input_buffer: Cell::new(Array2::<i8>::zeros((ITA::M as usize, ITA::M as usize))),
            _weight_buffer0: Cell::new(Array2::<i8>::zeros((ITA::M as usize, ITA::M as usize))),
            _weight_buffer1: Cell::new(Array2::<i8>::zeros((ITA::M as usize, ITA::M as usize))),
            _bias_buffer: Cell::new(Array1::<i32>::zeros(ITA::M as usize)),
            _output_buffer_i32: Cell::new(Array2::<i32>::zeros((ITA::M as usize, ITA::M as usize))),
            _output_buffer_i32_prev: Cell::new(Array2::<i32>::zeros((
                ITA::M as usize,
                ITA::M as usize,
            ))),
            _output_buffer_i8: Cell::new(Array2::<i8>::zeros((ITA::M as usize, ITA::M as usize))),

            // Internal state
            _step: AtomicU32::new(ITAStep::IDLE as u32),
            _inner_tile: AtomicU32::new(0),
        }
    }
}

impl Peripheral for HWPEITA {
    fn get_name(&self) -> &'static str {
        "hwpe-ita"
    }

    // Store to ITA
    fn store(&self, cpu: &Cpu, addr: u32, value: u32, _mask: u32, _size: u8) {
        match addr {
            0x00 => unsafe {
                self.trigger.store(value as u32, Ordering::SeqCst);
                info!("[ITA, CPU {}] Store state 0x{:02x}", &cpu.hartid, value);
                // Out addresses are currently hardcoded in ITA
                // Start ITA
                cpu.engine.hwpe_busy.store(1, Ordering::SeqCst);
                // Insert stalls to cpu
                self.compute_tile(cpu);
                cpu.engine.hwpe_busy.store(0, Ordering::SeqCst);
                cpu.engine.hwpe_evt.store(1, Ordering::SeqCst);

                info!("[ITA, CPU {}] Done.", &cpu.hartid);
            },
            0x04 => {
                self.acquire.store(value as u32, Ordering::SeqCst);
                info!("[ITA, CPU {}] Store acquire 0x{:08x}", &cpu.hartid, value)
            }
            0x08 => {
                self.evt_enable.store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store evt_enable 0x{:08x}",
                    &cpu.hartid, value
                )
            }
            0x0C => {
                self.status.store(value as u32, Ordering::SeqCst);
                info!("[ITA, CPU {}] Store status 0x{:016x}", &cpu.hartid, value)
            }
            0x10 => {
                self.running_jobs.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store running_jobs 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x14 => {
                self.soft_clear.store(value, Ordering::SeqCst);
                cpu.engine.hwpe_evt.store(0, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store soft_clear 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            // 0x18 is reserved for future use
            0x20 => {
                self.input_ptr.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store input_ptr 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x24 => {
                self.weight_ptr0.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store weight_ptr0 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x28 => {
                self.weight_ptr1.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store weight_ptr1 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x2C => {
                self.bias_ptr.store(value, Ordering::SeqCst);
                info!("[ITA, CPU {}] Store bias_ptr 0x{:016x}", &cpu.hartid, value)
            }
            0x30 => {
                self.output_ptr.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store output_ptr 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x34 => {
                self.seq_len.store(value, Ordering::SeqCst);
                info!("[ITA, CPU {}] Store seq_len 0x{:016x}", &cpu.hartid, value)
            }
            0x38 => {
                self.ita_tiles.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store ita_tiles 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x3C => {
                self.eps_mult1.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store eps_mult1 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x40 => {
                self.eps_mult2.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store eps_mult2 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x44 => {
                self.eps_shift1.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store eps_shift1 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x48 => {
                self.eps_shift2.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store eps_shift2 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            0x4C => {
                self.eps_add1.store(value as i32, Ordering::SeqCst);
                info!("[ITA, CPU {}] Store eps_add1 0x{:016x}", &cpu.hartid, value)
            }
            0x50 => {
                self.eps_add2.store(value as i32, Ordering::SeqCst);
                info!("[ITA, CPU {}] Store eps_add2 0x{:016x}", &cpu.hartid, value)
            }
            0x54 => {
                self.ctrl_stream.store(value, Ordering::SeqCst);
                info!(
                    "[ITA, CPU {}] Store ctrl_stream 0x{:016x}",
                    &cpu.hartid, value
                )
            }
            _ => unimplemented!(),
        }
    }

    // Load from ITA
    fn load(&self, _cpu: &Cpu, addr: u32, _size: u8) -> u32 {
        match addr {
            0x00 => self.trigger.load(Ordering::SeqCst),
            0x04 => self.acquire.load(Ordering::SeqCst),
            0x08 => self.evt_enable.load(Ordering::SeqCst),
            0x0C => self.status.load(Ordering::SeqCst),
            0x10 => self.running_jobs.load(Ordering::SeqCst),
            0x14 => self.soft_clear.load(Ordering::SeqCst),
            // 0x18 is reserved for future use
            0x20 => self.input_ptr.load(Ordering::SeqCst),
            0x24 => self.weight_ptr0.load(Ordering::SeqCst),
            0x28 => self.weight_ptr1.load(Ordering::SeqCst),
            0x2C => self.bias_ptr.load(Ordering::SeqCst),
            0x30 => self.output_ptr.load(Ordering::SeqCst),
            0x34 => self.seq_len.load(Ordering::SeqCst),
            0x38 => self.ita_tiles.load(Ordering::SeqCst),
            0x3C => self.eps_mult1.load(Ordering::SeqCst),
            0x40 => self.eps_mult2.load(Ordering::SeqCst),
            0x44 => self.eps_shift1.load(Ordering::SeqCst),
            0x48 => self.eps_shift2.load(Ordering::SeqCst),
            0x4C => self.eps_add1.load(Ordering::SeqCst) as u32,
            0x50 => self.eps_add2.load(Ordering::SeqCst) as u32,
            0x54 => self.ctrl_stream.load(Ordering::SeqCst),
            _ => unimplemented!(),
        }
    }
}

impl HWPEITA {
    // Get requantization parameters
    fn eps_mult(&self) -> [u8; 6] {
        let rqs_mult_w1 = u32::to_ne_bytes(self.eps_mult1.load(Ordering::SeqCst));
        let rqs_mult_w2 = u32::to_ne_bytes(self.eps_mult2.load(Ordering::SeqCst));

        [
            rqs_mult_w1[0],
            rqs_mult_w1[1],
            rqs_mult_w1[2],
            rqs_mult_w1[3],
            rqs_mult_w2[0],
            rqs_mult_w2[1],
        ]
    }

    fn eps_shift(&self) -> [u8; 6] {
        let rqs_shift_w1 = u32::to_ne_bytes(self.eps_shift1.load(Ordering::SeqCst));
        let rqs_shift_w2 = u32::to_ne_bytes(self.eps_shift2.load(Ordering::SeqCst));

        [
            rqs_shift_w1[0],
            rqs_shift_w1[1],
            rqs_shift_w1[2],
            rqs_shift_w1[3],
            rqs_shift_w2[0],
            rqs_shift_w2[1],
        ]
    }

    fn eps_add(&self) -> [i8; 6] {
        let rqs_add_w1 = i32::to_ne_bytes(self.eps_add1.load(Ordering::SeqCst)).map(|c| c as i8);
        let rqs_add_w2 = i32::to_ne_bytes(self.eps_add2.load(Ordering::SeqCst)).map(|c| c as i8);

        [
            rqs_add_w1[0],
            rqs_add_w1[1],
            rqs_add_w1[2],
            rqs_add_w1[3],
            rqs_add_w2[0],
            rqs_add_w2[1],
        ]
    }

    fn weight_preload(&self) -> bool {
        self.ctrl_stream.load(Ordering::SeqCst) & 0x1 != 0
    }

    fn weight_nextload(&self) -> bool {
        self.ctrl_stream.load(Ordering::SeqCst) & 0x2 != 0
    }

    fn tile_s(&self) -> u32 {
        self.ita_tiles.load(Ordering::SeqCst) & 0xF
    }

    fn tile_e(&self) -> u32 {
        (self.ita_tiles.load(Ordering::SeqCst) >> 4) & 0xF
    }

    fn tile_p(&self) -> u32 {
        (self.ita_tiles.load(Ordering::SeqCst) >> 8) & 0xF
    }

    unsafe fn compute_tile(&self, cpu: &Cpu) {
        let mut _step = self._step.load(Ordering::SeqCst);
        let mut _inner_tile = self._inner_tile.load(Ordering::SeqCst);
        let _eps_mult = self.eps_mult();
        let _eps_shift = self.eps_shift();
        let _eps_add = self.eps_add();

        fn local_to_global_addr(cpu: &Cpu, addr: u32) -> u32 {
            if addr <= cpu.engine.config.memory.tcdm.size {
                addr + cpu.engine.config.memory.tcdm.start
                    + cpu.engine.config.memory.tcdm.offset * cpu.cluster_id as u32
            } else {
                0
            }
        }

        let _input_ptr = local_to_global_addr(cpu, self.input_ptr.load(Ordering::SeqCst));
        let _weight_ptr0 = local_to_global_addr(cpu, self.weight_ptr0.load(Ordering::SeqCst));
        let _weight_ptr1 = local_to_global_addr(cpu, self.weight_ptr1.load(Ordering::SeqCst));
        let _bias_ptr = local_to_global_addr(cpu, self.bias_ptr.load(Ordering::SeqCst));
        let _output_ptr = local_to_global_addr(cpu, self.output_ptr.load(Ordering::SeqCst));

        let ita_m = ITA::M as usize;
        let mut _input_buffer = self
            ._input_buffer
            .replace(Array2::<i8>::zeros((ita_m, ita_m)));
        let mut _weight_buffer0 = self
            ._weight_buffer0
            .replace(Array2::<i8>::zeros((ita_m, ita_m)));
        let mut _weight_buffer1 = self
            ._weight_buffer1
            .replace(Array2::<i8>::zeros((ita_m, ita_m)));
        let mut _output_buffer_i32 = self
            ._output_buffer_i32
            .replace(Array2::<i32>::zeros((ita_m, ita_m)));
        let mut _output_buffer_i32_prev = self
            ._output_buffer_i32_prev
            .replace(Array2::<i32>::zeros((ITA::M as usize, ITA::M as usize)));
        let mut _output_buffer_i8 = self
            ._output_buffer_i8
            .replace(Array2::<i8>::zeros((ita_m, ita_m)));

        if _step == ITAStep::IDLE as u32 {
            // Preload weights for first iteration
            if self.weight_preload() {
                ITA::ita_load_2d(cpu, &mut _weight_buffer0, _weight_ptr0, ITA::M, ITA::M, 1);
            }
            _step = ITAStep::Q as u32;
            self._step.store(_step, Ordering::SeqCst);
        } else {
            _weight_buffer0 = _weight_buffer1.clone();
        }

        debug!("[ITA] Compute tile");
        debug!("      - Step: {:?}", ITAStep::try_from(_step).unwrap());
        debug!("      - Inner tile: {:?}", _inner_tile);
        debug!("      - Eps mult: {:?}", _eps_mult);
        debug!("      - Eps shift: {:?}", _eps_shift);
        debug!("      - Eps add: {:?}", _eps_add);
        debug!("      - Weight preload: {:?}", self.weight_preload());
        debug!("      - Weight nextload: {:?}", self.weight_nextload());
        debug!("      - Tile S: {:?}", self.tile_s());
        debug!("      - Tile E: {:?}", self.tile_e());
        debug!("      - Tile P: {:?}", self.tile_p());
        debug!("      - Input ptr: 0x{:08x}", _input_ptr);
        debug!("      - Weight ptr 0: 0x{:08x}", _weight_ptr0);
        debug!("      - Weight ptr 1: 0x{:08x}", _weight_ptr1);
        debug!("      - Bias ptr: 0x{:08x}", _bias_ptr);
        debug!("      - Output ptr: 0x{:08x}", _output_ptr);

        if _step == ITAStep::Q as u32 {
            _weight_buffer0 = ITA::transpose_2d_arrays(&mut _weight_buffer0);
        }

        ITA::ita_load_2d(cpu, &mut _input_buffer, _input_ptr, ITA::M, ITA::M, 1);

        // Preload weights for next iteration
        if self.weight_nextload() {
            ITA::ita_load_2d(cpu, &mut _weight_buffer1, _weight_ptr1, ITA::M, ITA::M, 1);
            _weight_buffer1 = ITA::transpose_2d_arrays(&mut _weight_buffer1);
        }

        if (_step == ITAStep::Q as u32)
            || (_step == ITAStep::K as u32)
            || (_step == ITAStep::V as u32)
            || (_step == ITAStep::OW as u32)
        {
            let mut _bias_buffer = self._bias_buffer.replace(Array1::<i32>::zeros(ita_m));
            // Load biases
            ITA::ita_load_1d_i24(cpu, &mut _bias_buffer, _bias_ptr, ITA::M);
            if _step == ITAStep::V as u32 {
                _weight_buffer0 = ITA::transpose_2d_arrays(&mut _weight_buffer0);
                _input_buffer = ITA::transpose_2d_arrays(&mut _input_buffer);
                ITA::projection_space_transformation(
                    &mut _output_buffer_i32,
                    &mut _weight_buffer0,
                    &mut _input_buffer,
                    &mut _bias_buffer,
                    1,
                );
            } else {
                ITA::projection_space_transformation(
                    &mut _output_buffer_i32,
                    &mut _input_buffer,
                    &mut _weight_buffer0,
                    &mut _bias_buffer,
                    1,
                );
            }
            self._bias_buffer.set(_bias_buffer);
        } else if _step == ITAStep::QK as u32 {
            _weight_buffer0 = ITA::transpose_2d_arrays(&mut _weight_buffer0);
            ITA::query_key_correlation(
                &mut _input_buffer,
                &mut _weight_buffer0,
                &mut _output_buffer_i32,
            );
        } else if _step == ITAStep::AV as u32 {
            ITA::streaming_partial_softmax(
                &mut _input_buffer,
                &mut _output_buffer_i32,
                ITA::M,
                ITA::N,
            );
            ITA::single_head_computation(
                &mut _output_buffer_i32.clone(),
                &mut _weight_buffer0,
                &mut _output_buffer_i32,
            );
        }

        _output_buffer_i32 = _output_buffer_i32 + _output_buffer_i32_prev;
        _output_buffer_i32_prev = _output_buffer_i32.clone();

        _inner_tile += 1;
        self._inner_tile.store(_inner_tile, Ordering::SeqCst);
        let mut writeback_output = false;

        match self._step.load(Ordering::SeqCst).try_into() {
            Ok(ITAStep::Q) => {
                if _inner_tile == self.tile_e() {
                    self._step.store(ITAStep::K as u32, Ordering::SeqCst);
                    writeback_output = true;
                }
            }
            Ok(ITAStep::K) => {
                if _inner_tile == self.tile_e() {
                    self._step.store(ITAStep::V as u32, Ordering::SeqCst);
                    writeback_output = true;
                }
            }
            Ok(ITAStep::V) => {
                if _inner_tile == self.tile_p() {
                    self._step.store(ITAStep::QK as u32, Ordering::SeqCst);
                    writeback_output = true;
                }
            }
            Ok(ITAStep::QK) => {
                if _inner_tile == self.tile_s() {
                    self._step.store(ITAStep::AV as u32, Ordering::SeqCst);
                    writeback_output = true;
                }
            }
            Ok(ITAStep::AV) => {
                if _inner_tile == self.tile_e() {
                    self._step.store(ITAStep::OW as u32, Ordering::SeqCst);
                    writeback_output = true;
                }
            }
            Ok(ITAStep::OW) => {
                if _inner_tile == self.tile_p() {
                    self._step.store(ITAStep::IDLE as u32, Ordering::SeqCst);
                    writeback_output = true;
                }
            }
            Ok(ITAStep::IDLE) => {
                self._step.store(ITAStep::Q as u32, Ordering::SeqCst);
            }
            Err(_) => (),
        }

        if writeback_output {
            ITA::requantization_2d(
                &mut _output_buffer_i32,
                &mut _output_buffer_i8,
                _eps_mult[_step as usize],
                _eps_shift[_step as usize],
                _eps_add[_step as usize],
            );
            if _step == ITAStep::V as u32 {
                _output_buffer_i8 = ITA::transpose_2d_arrays(&mut _output_buffer_i8);
            }
            trace!("Output Buffer:");
            trace!("{}", _output_buffer_i32);
            trace!("Output Buffer (RQS):");
            trace!("{}", _output_buffer_i8);
            ITA::ita_store_2d(cpu, &mut _output_buffer_i8, _output_ptr, ITA::M, ITA::M, 1);

            self._inner_tile.store(0, Ordering::SeqCst);
            // Reset output buffer
            _output_buffer_i32_prev = Array2::<i32>::zeros((ITA::M as usize, ITA::M as usize));
        }

        // Update internal buffers
        self._input_buffer.set(_input_buffer);
        self._weight_buffer0.set(_weight_buffer0);
        self._weight_buffer1.set(_weight_buffer1);
        self._output_buffer_i32.set(_output_buffer_i32);
        self._output_buffer_i32_prev.set(_output_buffer_i32_prev);
        self._output_buffer_i8.set(_output_buffer_i8);
    }
}

#[derive(Default)]
struct MemPoolITA {
    state: [AtomicU32; 4],
    start_addr: [AtomicU32; 4],
    out_addr: [AtomicU32; 4],
    rqs_addr: [AtomicU32; 4],
    seq_len: [AtomicU32; 4],
    emb_len: [AtomicU32; 4],
    proj_len: [AtomicU32; 4],
}
impl Peripheral for MemPoolITA {
    /// should return the same name as in the config file
    fn get_name(&self) -> &'static str {
        "mempool-ita"
    }

    /// store instruction
    fn store(&self, cpu: &Cpu, addr: u32, value: u32, _mask: u32, _size: u8) {
        let i = addr as usize / 0x30;
        let addr = addr as usize % 0x30;

        match addr {
            0x00 => unsafe {
                self.state[i].store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store state 0x{:02x}",
                    i, &cpu.hartid, value
                );
                // Out addresses are currently hardcoded in ITA
                let mut return_value = value;
                if value & 0x1 == 1 {
                    // Start ITA
                    MemPoolITA::run_ita(
                        cpu,
                        self.start_addr[i].load(Ordering::SeqCst),
                        // All ITA cores fetch the Q and K vector always from the address specified to core 0
                        self.start_addr[0].load(Ordering::SeqCst),
                        self.out_addr[i].load(Ordering::SeqCst),
                        self.rqs_addr[i].load(Ordering::SeqCst),
                        self.seq_len[i].load(Ordering::SeqCst),
                        self.emb_len[i].load(Ordering::SeqCst),
                        self.proj_len[i].load(Ordering::SeqCst),
                        16,
                    );
                    // Set busy flag
                    return_value |= 0x2;
                    // Clear start flag
                    return_value &= !0x1;

                    self.state[i].store(return_value, Ordering::SeqCst);
                    info!("[ITA {}, CPU {}] Done.", i, &cpu.hartid);
                    info!(
                        "[ITA {}, CPU {}] Store state 0x{:02x}",
                        i, &cpu.hartid, return_value
                    );
                }
            },
            0x04 => {
                self.start_addr[i].store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store start address 0x{:08x}",
                    i, &cpu.hartid, value
                )
            }
            0x08 => {
                self.out_addr[i].store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store out address 0x{:08x}",
                    i, &cpu.hartid, value
                )
            }
            0x0C => {
                self.rqs_addr[i].store(value as u32, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store rqs_addr 0x{:016x}",
                    i, &cpu.hartid, value
                )
            }
            0x10 => {
                self.seq_len[i].store(value, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store seq_len 0x{:016x}",
                    i, &cpu.hartid, value
                )
            }
            0x14 => {
                self.emb_len[i].store(value, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store emb_len 0x{:016x}",
                    i, &cpu.hartid, value
                )
            }
            0x18 => {
                self.proj_len[i].store(value, Ordering::SeqCst);
                info!(
                    "[ITA {}, CPU {}] Store proj_len 0x{:016x}",
                    i, &cpu.hartid, value
                )
            }
            _ => unimplemented!(),
        }
    }
    /// load instruction
    fn load(&self, cpu: &Cpu, addr: u32, _size: u8) -> u32 {
        let i = addr as usize / 0x30;
        let addr = addr as usize % 0x30;

        match addr {
            0x00 => {
                let state = self.state[i].load(Ordering::SeqCst) & 0xFF;
                info!(
                    "[ITA {}, CPU {}] Read state 0x{:02x}",
                    i, &cpu.hartid, state
                );

                let busy_flag = state & 0x02;

                // WIESEP: As we have no timing model, just set the done flag if the busy flag is set
                if busy_flag == 0x02 {
                    let mut new_state = state;
                    // Clear the busy flag
                    new_state &= !0x02;

                    // Set the done flag
                    new_state |= 0x4;
                    self.state[i].store(new_state, Ordering::SeqCst);
                    info!(
                        "[ITA {}, CPU {}] > ITA is done -> store state 0x{:02x}",
                        i, &cpu.hartid, new_state
                    );
                }
                state
            }
            0x04 => self.start_addr[i].load(Ordering::SeqCst),
            0x08 => self.out_addr[i].load(Ordering::SeqCst),
            0x0C => self.rqs_addr[i].load(Ordering::SeqCst),
            0x10 => self.seq_len[i].load(Ordering::SeqCst),
            0x14 => self.emb_len[i].load(Ordering::SeqCst),
            0x18 => self.proj_len[i].load(Ordering::SeqCst),
            _ => unimplemented!(),
        }
    }
}

impl MemPoolITA {
    unsafe fn run_ita(
        cpu: &Cpu,
        start_address: u32,
        start_address_core0: u32,
        out_address: u32,
        rqs_address: u32,
        seq_len: u32,
        emb_len: u32,
        proj_len: u32,
        processing_engines: u32,
    ) {
        // Setup of matrices for query_projection_space_transformation and key_projection_space_transformation
        // Sequence of addresses are hardcoded
        let start = start_address;
        let w_o_addr = start;
        let w_v_addr = start + proj_len * emb_len;
        let w_k_addr = start + proj_len * emb_len * 2;
        let w_q_addr = start + proj_len * emb_len * 3 + seq_len * emb_len * 2;
        let b_o_addr = start + proj_len * emb_len * 4 + seq_len * emb_len * 2;
        let b_v_addr = start + proj_len * emb_len * 4 + seq_len * emb_len * 2 + emb_len * 4; // 32 bit biases
        let b_k_addr =
            start + proj_len * emb_len * 4 + seq_len * emb_len * 2 + emb_len * 4 + proj_len * 4; // 32 bit biases
        let b_q_addr =
            start + proj_len * emb_len * 4 + seq_len * emb_len * 2 + emb_len * 4 + proj_len * 8; // 32 bit biases

        let q_addr = start_address_core0 + proj_len * emb_len * 3;
        let k_addr = start_address_core0 + proj_len * emb_len * 3 + seq_len * emb_len;

        let mult_address = cpu.binary_load(rqs_address + 0x00, 2);
        let shift_address = cpu.binary_load(rqs_address + 0x04, 2);
        let add_address = cpu.binary_load(rqs_address + 0x08, 2);

        let rqs_mult_w1 = u32::to_ne_bytes(cpu.binary_load(mult_address + 0x00, 2));
        let rqs_mult_w2 = u32::to_ne_bytes(cpu.binary_load(mult_address + 0x04, 2));

        let rqs_mult: [u8; 6] = [
            rqs_mult_w1[0],
            rqs_mult_w1[1],
            rqs_mult_w1[2],
            rqs_mult_w1[3],
            rqs_mult_w2[0],
            rqs_mult_w2[1],
        ];

        let rqs_shift_w1 = u32::to_ne_bytes(cpu.binary_load(shift_address + 0x00, 2));
        let rqs_shift_w2 = u32::to_ne_bytes(cpu.binary_load(shift_address + 0x04, 2));

        let rqs_shift: [u8; 6] = [
            rqs_shift_w1[0],
            rqs_shift_w1[1],
            rqs_shift_w1[2],
            rqs_shift_w1[3],
            rqs_shift_w2[0],
            rqs_shift_w2[1],
        ];

        let rqs_add_w1 = u32::to_ne_bytes(cpu.binary_load(add_address + 0x00, 2)).map(|c| c as i8);
        let rqs_add_w2 = u32::to_ne_bytes(cpu.binary_load(add_address + 0x04, 2)).map(|c| c as i8);

        let rqs_add: [i8; 6] = [
            rqs_add_w1[0],
            rqs_add_w1[1],
            rqs_add_w1[2],
            rqs_add_w1[3],
            rqs_add_w2[0],
            rqs_add_w2[1],
        ];

        debug!("[ITA, CPU {}] w_o_addr 0x{:x}", &cpu.hartid, w_o_addr);
        debug!("[ITA, CPU {}] w_v_addr 0x{:x}", &cpu.hartid, w_v_addr);
        debug!("[ITA, CPU {}] w_k_addr 0x{:x}", &cpu.hartid, w_k_addr);
        debug!("[ITA, CPU {}] q_addr   0x{:x}", &cpu.hartid, q_addr);
        debug!("[ITA, CPU {}] k_addr   0x{:x}", &cpu.hartid, k_addr);
        debug!("[ITA, CPU {}] w_q_addr 0x{:x}", &cpu.hartid, w_q_addr);
        debug!("[ITA, CPU {}] b_o_addr 0x{:x}", &cpu.hartid, b_o_addr);
        debug!("[ITA, CPU {}] b_v_addr 0x{:x}", &cpu.hartid, b_v_addr);
        debug!("[ITA, CPU {}] b_k_addr 0x{:x}", &cpu.hartid, b_k_addr);
        debug!("[ITA, CPU {}] b_q_addr 0x{:x}", &cpu.hartid, b_q_addr);

        debug!(
            "[ITA, CPU {}] mult_address  0x{:x}",
            &cpu.hartid, mult_address
        );
        debug!(
            "[ITA, CPU {}] shift_address 0x{:x}",
            &cpu.hartid, shift_address
        );
        debug!(
            "[ITA, CPU {}] add_address   0x{:x}",
            &cpu.hartid, add_address
        );

        let split_e = emb_len / processing_engines;
        let split_p = proj_len / processing_engines;

        debug!(
            "[ITA, CPU {}] Start Address 0x{:x}, Out Address 0x{:x}",
            &cpu.hartid, start, out_address
        );
        debug!("[ITA, CPU {}] RQS Mult {:?}", &cpu.hartid, rqs_mult);
        debug!("[ITA, CPU {}] RQS Shift {:?}", &cpu.hartid, rqs_shift);
        debug!("[ITA, CPU {}] RQS Add {:?}", &cpu.hartid, rqs_add);
        debug!(
            "[ITA, CPU {}] S {:?}, E {:?}, P {:?}",
            &cpu.hartid, seq_len, emb_len, proj_len
        );
        debug!(
            "[ITA, CPU {}] Split E {:?}, Split P {:?}",
            &cpu.hartid, split_e, split_p
        );

        let mut q = Array2::<i8>::zeros((seq_len as usize, emb_len as usize));
        ITA::ita_load_2d(cpu, &mut q, q_addr, seq_len, emb_len, split_e);
        debug!("[ITA, CPU {}] q.shape: {:?}", &cpu.hartid, q.shape());
        debug!("[ITA, CPU {}] q: {}", &cpu.hartid, q);

        let mut w_q = Array2::<i8>::zeros((proj_len as usize, emb_len as usize));
        ITA::ita_load_2d(cpu, &mut w_q, w_q_addr, proj_len, emb_len, split_e);
        w_q = ITA::transpose_2d_arrays(&mut w_q);
        debug!("[ITA, CPU {}] w_q.shape: {:?}", &cpu.hartid, w_q.shape());
        debug!("[ITA, CPU {}] w_q: {}", &cpu.hartid, w_q);

        let mut k = Array2::<i8>::zeros((seq_len as usize, emb_len as usize));
        ITA::ita_load_2d(cpu, &mut k, k_addr, seq_len, emb_len, split_e);
        debug!("[ITA, CPU {}] k.shape: {:?}", &cpu.hartid, k.shape());
        debug!("[ITA, CPU {}] k: {}", &cpu.hartid, k);

        let mut w_k = Array2::<i8>::zeros((proj_len as usize, emb_len as usize));
        ITA::ita_load_2d(cpu, &mut w_k, w_k_addr, proj_len, emb_len, 1);
        w_k = ITA::transpose_2d_arrays(&mut w_k);
        debug!("[ITA, CPU {}] w_k.shape: {:?}", &cpu.hartid, w_k.shape());
        debug!("[ITA, CPU {}] w_k: {}", &cpu.hartid, w_k);

        // Setup of matrices for value_projection_space_transformation
        let mut b_v = Array1::<i32>::zeros(proj_len as usize);
        ITA::ita_load_1d_i32(cpu, &mut b_v, b_v_addr, 1, proj_len);
        debug!("[ITA, CPU {}] b_v.shape: {:?}", &cpu.hartid, b_v.shape());
        debug!("[ITA, CPU {}] b_v: {}", &cpu.hartid, b_v);

        let mut v = k.clone();
        let mut w_v = Array2::<i8>::zeros((proj_len as usize, emb_len as usize));
        ITA::ita_load_2d(cpu, &mut w_v, w_v_addr, proj_len, emb_len, 1);
        w_v = ITA::transpose_2d_arrays(&mut w_v);
        debug!("[ITA, CPU {}] w_v.shape: {:?}", &cpu.hartid, w_v.shape());
        debug!("[ITA, CPU {}] w_v: {}", &cpu.hartid, w_v);

        let mut v_p = Array2::<i32>::zeros((seq_len as usize, proj_len as usize));

        // matrices in the query_projection_space_transformation
        let mut b_q = Array1::<i32>::zeros(proj_len as usize);
        ITA::ita_load_1d_i32(cpu, &mut b_q, b_q_addr, 1, proj_len);
        debug!("[ITA, CPU {}] b_q.shape: {:?}", &cpu.hartid, b_q.shape());
        debug!("[ITA, CPU {}] b_q: {}", &cpu.hartid, b_q);
        let mut q_p = Array2::<i32>::zeros((seq_len as usize, proj_len as usize));

        // matrices in the key_projection_space_transformation
        let mut b_k = Array1::<i32>::zeros(proj_len as usize);
        ITA::ita_load_1d_i32(cpu, &mut b_k, b_k_addr, 1, proj_len);
        debug!("[ITA, CPU {}] b_k.shape: {:?}", &cpu.hartid, b_k.shape());
        debug!("[ITA, CPU {}] b_k: {}", &cpu.hartid, b_k);

        let mut k_p = Array2::<i32>::zeros((seq_len as usize, proj_len as usize));

        // matrices in the streaming_partial_softmax
        let mut a_requant = Array2::<i8>::zeros((seq_len as usize, seq_len as usize));
        let mut a_partial_softmax = Array2::<i32>::zeros((seq_len as usize, seq_len as usize));

        // matrices in multi_head_computation
        let mut out = Array2::<i32>::zeros((seq_len as usize, emb_len as usize));
        let mut b_o = Array1::<i32>::zeros(emb_len as usize);
        ITA::ita_load_1d_i32(cpu, &mut b_o, b_o_addr, 1, emb_len);

        debug!("[ITA, CPU {}] b_o.shape: {:?}", &cpu.hartid, b_o.shape());
        debug!("[ITA, CPU {}] b_o: {}", &cpu.hartid, b_o);

        let mut w_o = Array2::<i8>::zeros((emb_len as usize, proj_len as usize));
        ITA::ita_load_2d(cpu, &mut w_o, w_o_addr, emb_len, proj_len, 1);
        w_o = ITA::transpose_2d_arrays(&mut w_o);
        debug!("[ITA, CPU {}] w_o.shape: {:?}", &cpu.hartid, w_o.shape());
        debug!("[ITA, CPU {}] w_o: {}", &cpu.hartid, w_o);

        // query_projection_space_transformation
        ITA::projection_space_transformation(&mut q_p, &mut q, &mut w_q, &mut b_q, 1);
        // requantization of q_p
        let mut q_p_requant = Array2::<i8>::zeros((seq_len as usize, proj_len as usize));
        ITA::requantization_2d(
            &mut q_p,
            &mut q_p_requant,
            rqs_mult[0],
            rqs_shift[0],
            rqs_add[0],
        );
        debug!("[ITA, CPU {}] q_p_requant: {}", &cpu.hartid, q_p_requant);

        // key_projection_space_transformation
        ITA::projection_space_transformation(&mut k_p, &mut k, &mut w_k, &mut b_k, 1);
        // requantization of k_p
        let mut k_p_requant = Array2::<i8>::zeros((seq_len as usize, proj_len as usize));
        ITA::requantization_2d(
            &mut k_p,
            &mut k_p_requant,
            rqs_mult[1],
            rqs_shift[1],
            rqs_add[1],
        );
        debug!("[ITA, CPU {}] k_p_requant: {}", &cpu.hartid, k_p_requant);

        // query_key_correlation
        let mut qk = Array2::<i32>::zeros((seq_len as usize, seq_len as usize));
        ITA::query_key_correlation(&mut q_p_requant, &mut k_p_requant, &mut qk);
        // requantization of qk
        ITA::requantization_2d(
            &mut qk,
            &mut a_requant,
            rqs_mult[2],
            rqs_shift[2],
            rqs_add[2],
        );
        debug!("[ITA, CPU {}] a_requant: {}", &cpu.hartid, a_requant);

        // streaming_partial_softmax
        ITA::streaming_partial_softmax(
            &mut a_requant,
            &mut a_partial_softmax,
            seq_len,
            processing_engines,
        );

        // value_projection_space_transformation
        ITA::projection_space_transformation(&mut v_p, &mut v, &mut w_v, &mut b_v, 1);
        // requantization of v_p
        let mut v_p_requant = Array2::<i8>::zeros((seq_len as usize, proj_len as usize));
        ITA::requantization_2d(
            &mut v_p,
            &mut v_p_requant,
            rqs_mult[3],
            rqs_shift[3],
            rqs_add[3],
        );
        debug!("[ITA, CPU {}] v_p_requant: {}", &cpu.hartid, v_p_requant);

        // single_head_computation
        let mut o_softmax = Array2::<i32>::zeros((seq_len as usize, proj_len as usize));
        ITA::single_head_computation(&mut a_partial_softmax, &mut v_p_requant, &mut o_softmax);
        // requantization of o_softmax
        let mut o_softmax_requant = Array2::<i8>::zeros((seq_len as usize, proj_len as usize));
        ITA::requantization_2d(
            &mut o_softmax,
            &mut o_softmax_requant,
            rqs_mult[4],
            rqs_shift[4],
            rqs_add[4],
        );
        debug!(
            "[ITA, CPU {}] o_softmax_requant: {}",
            &cpu.hartid, o_softmax_requant
        );

        // multi_head_computation
        ITA::multi_head_computation(&mut o_softmax_requant, &mut out, &mut w_o, &mut b_o, 1);
        // parallel requantization of out
        let mut out_requant = Array2::<i8>::zeros((seq_len as usize, emb_len as usize));
        ITA::requantization_2d(
            &mut out,
            &mut out_requant,
            rqs_mult[5],
            rqs_shift[5],
            rqs_add[5],
        );
        debug!("[ITA, CPU {}] out_requant: {}", &cpu.hartid, out_requant);

        // Store the output
        ITA::ita_store_2d(cpu, &out_requant, out_address, seq_len, emb_len, 1);
    }
}

trait ITAConfig {
    const M: u32 = 64; // Tile size
    const N: u32 = 16; // Number of sumdotp units
}
struct ITA {}

impl Default for ITA {
    fn default() -> Self {
        ITA {}
    }
}

impl ITAConfig for ITA {}

impl ITA {
    fn transpose_2d_arrays<T>(array: &mut Array2<T>) -> Array2<T>
    where
        T: Clone,
    {
        return array.to_owned().permuted_axes([1, 0]);
    }

    unsafe fn ita_load_2d_i32(cpu: &Cpu, data: &mut Array2<i32>, mut address: u32, m: u32, n: u32) {
        for j in 0..m {
            for i in 0..n {
                let word = cpu.binary_load(address, 2);
                data[[j as usize, i as usize]] = word as i32;
                address += 4;
            }
        }
    }

    unsafe fn ita_load_1d_i32(cpu: &Cpu, data: &mut Array1<i32>, mut address: u32, m: u32, n: u32) {
        for i in 0..n {
            let word = cpu.binary_load(address, 2);
            data[[i as usize]] = word as i32;
            address += 4;
        }
    }

    unsafe fn ita_load_1d_i24(cpu: &Cpu, data: &mut Array1<i32>, mut address: u32, n: u32) {
        // Unpack 24 bit values packed to 32 bit words
        let mut shift = 0;
        let mut current_word = cpu.binary_load(address, 2);
        let mut next_word;

        let mut values_read = 0;

        while values_read < n {
            let mut value = (current_word >> shift) & 0x00FFFFFF;

            shift += 24;

            // Check if the next value will go beyond current word
            if shift >= 32 {
                // Load the next word
                address += 4; // Assuming the word size in memory is 4 bytes
                next_word = cpu.binary_load(address, 2);

                // Calculate the number of bits from the next word
                let bits_from_next = shift - 32;
                let extra_bits = (next_word & ((1 << bits_from_next) - 1)) << (24 - bits_from_next);

                // Combine the bits from current and next word
                value = value | extra_bits;

                // Reset shift and update the current word
                shift = bits_from_next;
                current_word = next_word;
            }

            // Apply sign extension if the 24th bit is set (sign bit for 24-bit integer)
            if value & 0x00800000 != 0 {
                value |= 0xFF000000;
            }
            data[[values_read as usize]] = value as i32;

            values_read += 1;
        }
    }

    unsafe fn ita_load_2d(
        cpu: &Cpu,
        data: &mut Array2<i8>,
        mut address: u32,
        m: u32,
        n: u32,
        splits: u32,
    ) {
        for split in 0..splits {
            for j in 0..m {
                for i in (0..n / splits).step_by(4) {
                    let word = cpu.binary_load(address, 2);
                    let elements = std::mem::transmute::<u32, [i8; 4]>(word);
                    for (offset, e) in elements.iter().enumerate() {
                        data[[j as usize, ((n / splits) * split + i) as usize + offset]] = *e;
                    }
                    address += 4;
                }
            }
        }
    }

    unsafe fn ita_load_3d(
        cpu: &Cpu,
        data: &mut Array3<i8>,
        mut address: u32,
        m: u32,
        n: u32,
        p: u32,
        splits: u32,
    ) {
        for split in 0..splits {
            for j in 0..m {
                for i in 0..n {
                    for h in (0..p / splits).step_by(4) {
                        let word = cpu.binary_load(address, 2);
                        let elements = std::mem::transmute::<u32, [i8; 4]>(word);
                        for (offset, e) in elements.iter().enumerate() {
                            data[[
                                j as usize,
                                i as usize,
                                ((p / splits) * split + h) as usize + offset,
                            ]] = *e;
                        }
                        address += 4;
                    }
                }
            }
        }
    }

    unsafe fn ita_store_2d(
        cpu: &Cpu,
        data: &Array2<i8>,
        address: u32,
        m: u32,
        n: u32,
        splits: u32,
    ) {
        let mut address_offset = 0;
        for split in 0..splits {
            for j in 0..m {
                for i in (0..n / splits).step_by(4) {
                    let mut elements = [0u8; 4];
                    for offset in 0..elements.len() {
                        elements[offset] =
                            data[[j as usize, ((n / splits) * split + i) as usize + offset]] as u8;
                    }
                    let word = u32::from_ne_bytes(elements);
                    cpu.binary_store(address + address_offset, word, u32::MAX, 2);
                    trace!(
                        "[ITA, CPU {}] Store OUT to 0x{:x}",
                        &cpu.hartid,
                        address + address_offset
                    );
                    address_offset += 4;
                }
            }
        }
    }

    fn requantize(element: i32, eps_mult: u8, right_shift: u8, add: i8) -> i8 {
        let mut shifted = ((element * (eps_mult as i32)) >> (right_shift as i32)) + (add as i32);

        // Perform rounding half away from zero
        if right_shift > 0
            && ((element * (eps_mult as i32)) >> ((right_shift - 1) as i32)) & 0x1 == 1
        {
            shifted = shifted.saturating_add(1);
        }
        if shifted > 127 {
            return 127;
        } else if shifted < -128 {
            return -128;
        } else {
            return shifted as i8;
        }
    }

    fn requantization_2d(
        m: &mut Array2<i32>,
        m_requant: &mut Array2<i8>,
        eps_mult: u8,
        right_shift: u8,
        add: i8,
    ) {
        // Loop over the head dimension
        for j in 0..m.shape()[0] {
            // print the column of the head matrix
            let row = m.slice(s![j, ..]);
            // Iterate over the row and requantize it
            for k in 0..row.len() {
                m_requant[[j, k]] = ITA::requantize(row[k], eps_mult, right_shift, add);
            }
        }
    }

    fn parallel_requantize3d(
        m: &mut Array3<i32>,
        m_requant: &mut Array2<i8>,
        eps_mult: u8,
        right_shift: u8,
        add: i32,
    ) {
        m_requant.fill(add as i8);
        for i in 0..m.shape()[0] {
            for j in 0..m.shape()[1] {
                let row = m.slice(s![i, j, ..]);
                for k in 0..row.len() {
                    let mut shifted = ((row[k] * (eps_mult as i32)) >> (right_shift as i32))
                        + m_requant[[i * m.shape()[1] + j, k]] as i32;

                    // Perform rounding half away from zero
                    if right_shift > 0
                        && ((row[k] * (eps_mult as i32)) >> ((right_shift - 1) as i32)) & 0x1 == 1
                    {
                        shifted = shifted.saturating_add(1);
                    }
                    m_requant[[i * m.shape()[1] + j, k]] = ITA::requantize(shifted, 1, 0, 0);
                }
            }
        }
    }

    fn projection_space_transformation(
        p: &mut Array2<i32>,
        m: &mut Array2<i8>,
        w: &mut Array2<i8>,
        b: &mut Array1<i32>,
        bias: u8,
    ) {
        info!("===================== Projection Space Transformation =====================");
        info!("p shape: {:?}", p.shape());
        info!("m shape: {:?}", m.shape());
        info!("w: {:?}", w.shape());
        info!("b: {:?}", b.shape());

        trace!("m: {}", m);
        trace!("w: {}", w);
        trace!("b: {}", b);

        // Calculate p[h] = m * W[h] + b[h] for each head h

        let d1 = m.shape();
        let d2 = w.shape();

        assert_eq!(d1[0], d2[1], "Matrices dimensions don't match");

        let slice_a = m.map(|x| *x as i32);
        let slice_b = w.map(|x| *x as i32);
        let slice_c = b.map(|x| *x);
        let slice_c = slice_c.broadcast((d1[0], d2[1])).unwrap().map(|x| *x);
        let mut mult_a_b = slice_a.dot(&slice_b);

        if bias == 1 {
            mult_a_b = mult_a_b + slice_c;
        }

        p.assign(&mult_a_b);
    }

    fn query_key_correlation(
        qp_requant: &mut Array2<i8>,
        kp_requant: &mut Array2<i8>,
        qk: &mut Array2<i32>,
    ) {
        info!("===================== Query Key Correlation =====================");
        info!("qp_requant shape: {:?}", qp_requant.shape());
        info!("kp_requant shape: {:?}", kp_requant.shape());

        trace!("qp_requant: {}", qp_requant);
        trace!("kp_requant: {}", kp_requant);

        let d1 = qp_requant.shape();
        let d2 = kp_requant.shape();

        assert_eq!(d1[1], d2[1], "Matrices dimensions don't match");

        // Calculate qk[h] = qp_requant[h] * kp_requant[h].T for each head h
        let kp_requant_transposed = ITA::transpose_2d_arrays(kp_requant);

        let slice_a = qp_requant.map(|x| *x as i32);
        let slice_b = kp_requant_transposed.map(|x| *x as i32);
        let mult_a_b = slice_a.dot(&slice_b);

        qk.assign(&mult_a_b);
    }

    //Compute the approximated softmax function.
    fn streaming_partial_softmax(
        a_requant: &mut Array2<i8>,
        a_partial_softmax: &mut Array2<i32>,
        seq_len: u32,
        processing_engines: u32,
    ) {
        // let log2e: f64 = f64::log2(f64::exp(1.0));
        // let b = 8;
        // let eps_x = b as f64 / (2.0f64.powi(b) * log2e);
        let mut exp_partial_sum = Array1::<i32>::zeros(seq_len as usize);
        let mut max = Array1::<i8>::zeros(seq_len as usize);
        let mut current_max = Array1::<i8>::zeros(seq_len as usize);
        let _processing_engines = processing_engines as usize;
        let groups = seq_len as usize / _processing_engines;

        for i in 0..groups {
            let a_requant_slice = a_requant.slice_mut(s![
                ..,
                i * _processing_engines..(i + 1) * _processing_engines
            ]);

            for n in 0..a_requant_slice.nrows() {
                current_max[[n]] = a_requant_slice.row(n).iter().copied().max().unwrap() as i8;
            }

            for j in 0..seq_len {
                let mut shift_sum: u8;
                if i == 0 || current_max[j as usize] > max[[j as usize]] {
                    if i == 0 {
                        shift_sum = 0;
                    } else {
                        let shift_int =
                            (current_max[j as usize] as i32) - (max[[j as usize]] as i32);
                        shift_sum = (shift_int / 32) as u8;

                        if shift_int % 32 >= 16 {
                            shift_sum += 1;
                        }
                    }
                    max[j as usize] = current_max[j as usize];
                } else {
                    shift_sum = 0;
                }

                let qb = a_requant
                    .slice_mut(s![
                        ..,
                        i * _processing_engines..(i + 1) * _processing_engines
                    ])
                    .mapv(|x| x as i32 - max[[j as usize]] as i32);

                let mut qexp = 0;
                for k in 0..qb.ncols() {
                    let mut shift = (-qb[[j as usize, k]]) as i32 / 32;
                    let shift_int = (-qb[[j as usize, k]]) as i32;

                    if shift_int % 32 >= 16 {
                        shift += 1;
                    }

                    qexp += (2_u32.pow(10) >> shift as i32) as i32;
                }

                exp_partial_sum[[j as usize]] =
                    (exp_partial_sum[[j as usize]] >> shift_sum as i32) + qexp;
            }
        }
        for j in 0..seq_len {
            let factor =
                ((2.0f64.powi(7) - 1.0) * 2.0f64.powi(10)) as i32 / exp_partial_sum[j as usize];
            for k in 0..seq_len {
                let mut shift = (((max[j as usize] as i32)
                    - (a_requant[[j as usize, k as usize]] as i32))
                    / 32) as i32;
                let shift_int =
                    (max[j as usize] as i32) - (a_requant[[j as usize, k as usize]] as i32);
                if shift_int % 32 >= 16 {
                    shift += 1;
                }
                a_partial_softmax[[j as usize, k as usize]] =
                    (factor as i32) / 2.0f64.powi(shift) as i32;
            }
        }
    }

    fn single_head_computation(
        a_partial_softmax: &mut Array2<i32>,
        vp_requant: &mut Array2<i8>,
        o_softmax: &mut Array2<i32>,
    ) {
        trace!("a_partial_softmax: {}", a_partial_softmax);
        trace!("vp_requant: {}", vp_requant);
        // Loop over the number of queries
        for j in 0..o_softmax.shape()[0] {
            // Loop over the number of keys
            for k in 0..o_softmax.shape()[1] {
                o_softmax[[j, k]] = 0;
                // Loop over the number of features
                for l in 0..o_softmax.shape()[0] {
                    o_softmax[[j, k]] +=
                        a_partial_softmax[[j, l]] as i32 * vp_requant[[l, k]] as i32;
                }
            }
        }
    }

    fn multi_head_computation(
        o_softmax_requant: &mut Array2<i8>,
        out: &mut Array2<i32>,
        w_o: &mut Array2<i8>,
        b_o: &mut Array1<i32>,
        bias: u8,
    ) {
        info!("===================== Multi Head Computation =====================");
        info!("o_softmax_requant shape: {:?}", o_softmax_requant.shape());
        info!("out shape: {:?}", out.shape());
        info!("w_o shape: {:?}", w_o.shape());
        info!("b_o shape: {:?}", b_o.shape());

        let d1 = o_softmax_requant.shape();
        let d2 = w_o.shape();

        assert_eq!(d1[2], d2[1], "Matrices dimensions don't match");

        // Calculate out[h] = o_softmax_requant[h] * W_o[h] + b_o[h] for each head h

        let slice_a = o_softmax_requant.map(|x| *x as i32);
        let slice_b = w_o.map(|x| *x as i32);
        let slice_c = b_o.map(|x| *x);
        let slice_c = slice_c.broadcast((d1[0], d2[2])).unwrap().map(|x| *x);
        let mut mult_a_b = slice_a.dot(&slice_b);

        if bias == 1 {
            mult_a_b = mult_a_b + slice_c;
        }

        out.assign(&mult_a_b);
    }
}
