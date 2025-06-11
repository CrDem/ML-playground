use cudarc::{
    driver::{CudaContext, CudaSlice, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

const SIZE: usize = 1024;

fn main() -> Result<(), DriverError> {

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let a = vec![1.0f32; SIZE];
    let b = vec![2.0f32; SIZE];

    let mut d_a: CudaSlice<f32> = stream.alloc_zeros::<f32>(SIZE)?;
    let mut d_b: CudaSlice<f32> = stream.alloc_zeros::<f32>(SIZE)?;
    let mut d_c: CudaSlice<f32> = stream.alloc_zeros::<f32>(SIZE)?;

    stream.memcpy_htod(&a, &mut d_a)?;
    stream.memcpy_htod(&b, &mut d_b)?;

    // You can load a function from a pre-compiled PTX like so:
    let module = ctx.load_module(Ptx::from_file("./CUDA kernels/vec_add.ptx"))?;

    // and then load a function from it:
    let func = module.load_function("vec_add").unwrap();

    // we use a buidler pattern to launch kernels.
    let n = a.len() as i32;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(&d_a);
    launch_args.arg(&d_b);
    launch_args.arg(&mut d_c);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }?;

    let c: Vec<f32> = stream.memcpy_dtov(&d_c)?;
    assert_eq!(c, [3.0f32; SIZE]);

    println!("a: {:?}", &a[..10]);
    println!("b: {:?}", &b[..10]);
    println!("c: {:?}", &c[..10]);

    Ok(())
}