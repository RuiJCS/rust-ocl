extern crate ocl;
extern crate image;

mod filter;
mod ocl_program;

use filter::{Square, Filter};
use ocl_program::OclProgram;

use std::fs::File;
use std::io::Read;
use std::path::Path;

use ocl::{Context, Queue, Device, Program, Image, Sampler, Kernel, Buffer, EventList};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, AddressingMode, FilterMode, MemObjectType};
use ocl::flags::{CommandQueueProperties};





// fn trivial() -> ocl::Result<()> {
//     // let src = r#"
//     //     __kernel void add(__global float* buffer, float scalar) {
//     //         buffer[get_global_id(0)] += scalar;
//     //     }
//     // "#;

//     let mut f = file::open("src/kernels.cl").expect("file not found");

//     let mut src = String::new();
//     f.read_to_string(&mut src)
//         .expect("something went wrong reading the file");

//     // println!("{}", src);

//     let pro_que = ProQue::builder()
//         .src(src)
//         .dims(64)
//         .build()?;

//     let data = pro_que.create_buffer::<f32>()?;
//     let result = pro_que.create_buffer::<f32>()?;

//     let kernel = pro_que.kernel_builder("add_numbers")
//         .arg_buf(&data)
//         .arg_buf(&result)
//         .build()?;
//         // .arg_scl(10.0f32);

//     unsafe { kernel.enq()?; }

//     let mut vec = vec![0.0f32; result.len()];
//     result.read(&mut vec).enq()?;

//     println!("The value at index [{}] is now '{}'!", index, vec[index]);
//     Ok(())
// }

fn paint_blue() -> ocl::Result<()> {const KERNEL_SIZE: u32 = 11;
    const KERNEL_SIZE_HALF: u32 = KERNEL_SIZE / 2;
    let mut img = image::open("files/leninha.jpg")
                .unwrap()
                .to_rgba();
    let dims = img.dimensions();
    let row_s = dims.1;
    let col_s = dims.0;
    let kernel_half = format!("-D KERNEL_SIZE_HALF={}",KERNEL_SIZE_HALF);
    let row_size = format!("-D ROW_SIZE={}",row_s);
    let col_size = format!("-D COL_SIZE={}",col_s);
    let cl_opts = format!("{} {} {}",kernel_half, row_size,col_size);
    let _index = 43;


    let mut f = File::open("src/kernels.cl").expect("file not found");

    let mut src = String::new();
    f.read_to_string(&mut src)
        .expect("something went wrong reading the file");

    let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src(src)
        .cmplr_opt(cl_opts)
        .devices(device)
        .build(&context).unwrap();


    let src_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_READ_ONLY | ocl::flags::MEM_HOST_WRITE_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(&img)
        .queue(queue.clone())
        .build().unwrap();

    let dst_image = Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(ocl::flags::MEM_WRITE_ONLY | ocl::flags::MEM_HOST_READ_ONLY | ocl::flags::MEM_COPY_HOST_PTR)
        .copy_host_slice(&img)
        .queue(queue.clone())
        .build().unwrap();

    let kernel = Kernel::builder()
        .program(&program)
        .name("blue")
        .queue(queue.clone())
        .gws(&dims)
        .arg_img(&src_image)
        .arg_img(&dst_image)
        .build().unwrap();

    unsafe { kernel.enq().unwrap(); }

    dst_image.read(&mut img).enq().unwrap();

    img.save(&Path::new("files/out_blue.jpg")).unwrap();

    Ok(())
}

fn test() {
    let mut ocl = OclProgram::new(5, "src/kernels.cl".to_string(), "convolute".to_string(), "leninha.jpg".to_string());
    ocl.run();
    println!("{}", ocl.print_profile());
}

fn main() {
    // trivial().unwrap();
    // let img = image::open("files/img.jpg").unwrap();

    // let ref mut fout = file::create("files/out.jpg").unwrap();

    // img.save(fout,image::JPEG).unwrap();
    // paint_blue().unwrap();
    // convolute().unwrap();

    test();

}