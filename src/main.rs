mod matrix;

use clap::{Arg, command, value_parser};
use log::info;
use matrix::Matrix;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use std::{fs::read, path::PathBuf};

use cudarc::{
    driver::{CudaContext, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

///////// локальная обёртка вокруг half::f16: /////////
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct F16(half::f16);

unsafe impl DeviceRepr for F16 {}

impl From<half::f16> for F16 {
    fn from(value: half::f16) -> Self {
        F16(value)
    }
}

impl From<F16> for half::f16 {
    fn from(value: F16) -> half::f16 {
        value.0
    }
}
///////////////////////////////////////////////////////

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let matches_result = command!()
        .arg(
            Arg::new("input_tensors_path")
                .long("input")
                .short('i')
                .value_parser(value_parser!(PathBuf))
                .help("Input tensor path relative to src directory")
                .required(true),
        )
        .arg(
            Arg::new("output_tensor_path")
                .long("output")
                .short('o')
                .value_parser(value_parser!(PathBuf))
                .help("Ouput path to write resulted tensor")
                .required(true),
        )
        .get_matches();

    let input_path = matches_result
        .get_one::<PathBuf>("input_tensors_path")
        .unwrap();
    let output_path = matches_result
        .get_one::<PathBuf>("output_tensor_path")
        .unwrap();

    info!("Reading tensors from file");
    let data = read(input_path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    info!("Parsing tensors to matrixes");
    let a_tensor = tensors.tensor("A")?;
    let b_tensor = tensors.tensor("B")?;

    let a = Matrix::<half::f16>::from_bytes(a_tensor.data(), a_tensor.shape());
    let b = Matrix::<half::f16>::from_bytes(b_tensor.data(), b_tensor.shape());

    // Конвертируем в обёрнутый тип
    let a_wrapped: Matrix<F16> = Matrix {
        data: a.data.into_iter().map(F16::from).collect(),
        size: a.size,
    };
    let b_wrapped: Matrix<F16> = Matrix {
        data: b.data.into_iter().map(F16::from).collect(),
        size: b.size,
    };

    let m = a.size.0 as u32;
    let n = b.size.1 as u32;
    let k = a.size.1 as u32;

    info!("Initialization cuda context");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    info!("Binding data");
    let mut d_a: CudaSlice<F16> = unsafe {stream.alloc(a.size.0 * a.size.1)?};
    let mut d_b: CudaSlice<F16> = unsafe {stream.alloc(a.size.1 * b.size.1)?};
    let mut d_c: CudaSlice<F16> = unsafe {stream.alloc(a.size.0 * b.size.1)?};

    stream.memcpy_htod(&a_wrapped.data, &mut d_a)?;
    stream.memcpy_htod(&b_wrapped.data, &mut d_b)?;

    info!("Loading PTX module");
    let module = ctx.load_module(Ptx::from_file("./shaders/matmul.ptx"))?;
    let func = module.load_function("gemm_kernel_fp16").unwrap();

    let threads_per_block = (32, 32, 1);
    let blocks_per_grid = (
        (n + threads_per_block.0 - 1) / threads_per_block.0,
        (m + threads_per_block.1 - 1) / threads_per_block.1,
        1
    );
    let cfg = LaunchConfig {
        grid_dim: blocks_per_grid,
        block_dim: threads_per_block,
        shared_mem_bytes: 0,
    };
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(&d_a);
    launch_args.arg(&d_b);
    launch_args.arg(&mut d_c);
    launch_args.arg(&m);
    launch_args.arg(&n);
    launch_args.arg(&k);

    info!("Launching kernel");
    let start = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;
    let stop = ctx.new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

    // Launch the kernel a few times to avoid cold start
    for _ in 0..5 {
        unsafe { launch_args.launch(cfg) }?;
    }

    start.record(&stream)?;
    unsafe { launch_args.launch(cfg) }?;
    stop.record(&stream)?;

    stop.synchronize()?;

    let elapsed_ms = start.elapsed_ms(&stop)? as f64;
    let total_ops = (2 * m * n * k - m * n) as f64;
    let gflops = total_ops / elapsed_ms / 1e6;
    println!("Computation took {} ms\nGFLOPS: {}", elapsed_ms, gflops);

    info!("Copying result back");
    let mut c = vec![F16(half::f16::ZERO); a.size.0 * b.size.1];
    stream.memcpy_dtoh(&d_c, &mut c)?;

    // Конвертируем обратно в f16 для сохранения
    let result_data: Vec<u8> = c.into_iter()
        .flat_map(|f| f.0.to_le_bytes())
        .collect();

    let result_shape = vec![a.size.0, b.size.1];
    let tensor_view = TensorView::new(Dtype::F16, result_shape.clone(), &result_data)?;

    info!("Saving results");
    safetensors::serialize_to_file(
        [("C".to_string(), tensor_view)].into_iter(),
        None,
        output_path,
    )?;

    /*dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);*/

    /*info!("Initialization metal context");
    let mut gemm_core = GeMMMetalContext::gemm_init();

    info!("Binding data");
    gemm_core.bind_data(&mut a, &mut b);

    info!("Computing");
    let start = Instant::now();
    let result_data = unsafe { gemm_core.compute() };
    let duration = start.elapsed();
    println!("{} took {:?}", "Computation", duration);

    let result_shape = vec![a.size.0, b.size.1];
    let result_data = unsafe {
        std::slice::from_raw_parts(
            result_data.as_ptr() as *const u8,
            result_shape.iter().product::<usize>() * std::mem::size_of::<f32>(),
        )
    };

    let tensor_view = TensorView::new(Dtype::F32, result_shape, result_data)?; //fp16

    safetensors::serialize_to_file(
        [("C".to_string(), tensor_view)].into_iter(),
        None,
        &output_path,
    )?;*/

    Ok(())
}
