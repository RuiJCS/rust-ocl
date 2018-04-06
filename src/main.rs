extern crate ocl;
extern crate image;
#[macro_use] extern crate colorify;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use ocl::ProQue;
use image::GenericImage;

use ocl::{Context, Queue, Device, Program, Image, Sampler, Kernel, Buffer};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, AddressingMode, FilterMode, MemObjectType};




fn trivial() -> ocl::Result<()> {
    let index = 43;
    // let src = r#"
    //     __kernel void add(__global float* buffer, float scalar) {
    //         buffer[get_global_id(0)] += scalar;
    //     }
    // "#;

    let mut f = File::open("src/kernels.cl").expect("file not found");

    let mut src = String::new();
    f.read_to_string(&mut src)
        .expect("something went wrong reading the file");

    // println!("{}", src);

    let pro_que = ProQue::builder()
        .src(src)
        .dims(64)
        .build()?;

    let data = pro_que.create_buffer::<f32>()?;
    let result = pro_que.create_buffer::<f32>()?;

    let kernel = pro_que.kernel_builder("add_numbers")
        .arg_buf(&data)
        .arg_buf(&result)
        .build()?;
        // .arg_scl(10.0f32);

    unsafe { kernel.enq()?; }

    let mut vec = vec![0.0f32; result.len()];
    result.read(&mut vec).enq()?;

    println!("The value at index [{}] is now '{}'!", index, vec[index]);
    Ok(())
}

fn paint_blue() -> ocl::Result<()> {

    let mut img = image::open("files/leninha.jpg")
                .unwrap()
                .to_rgba();

    let mut f = File::open("src/kernels.cl").expect("file not found");

    let mut src = String::new();
    f.read_to_string(&mut src)
        .expect("something went wrong reading the file");

    let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src(src)
        .cmplr_opt("-D KERNEL_SIZE_HALF=2")
        .devices(device)
        .build(&context).unwrap();

    let sup_img_formats = Image::<u8>::supported_formats(&context, ocl::flags::MEM_READ_WRITE,
        MemObjectType::Image2d).unwrap();
    println!("Image formats supported: {}.", sup_img_formats.len());

    let dims = img.dimensions();

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

    let sampler = Sampler::new(&context, true, AddressingMode::None, FilterMode::Nearest).unwrap();

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

    // println!("{:?}",img);

    // let ref mut fout = File::create("files/out.jpg").unwrap();

    img.save(&Path::new("files/out.jpg")).unwrap();

    Ok(())
}


fn convolute() -> ocl::Result<()> {
    const KERNEL_SIZE: u32 = 501;
    const KERNEL_SIZE_HALF: u32 = KERNEL_SIZE / 2;
    const BUFF_SIZE: u32 = KERNEL_SIZE * KERNEL_SIZE;
    const BUFF_VAL: f32 = 1.0 / BUFF_SIZE as f32;
    let FILE: String = String::from("leninha.jpg");
    let INPUT_FILE: String = format!("files/{}",FILE);
    let OUTPUT_FILE: String = format!("files/output_{}",FILE);
    let kernel_half: String = format!("-D KERNEL_SIZE_HALF={}",KERNEL_SIZE_HALF);

    let mut img = image::open(INPUT_FILE)
                .unwrap()
                .to_rgba();

    let mut f = File::open("src/kernels.cl").expect("file not found");

    let mut src = String::new();
    f.read_to_string(&mut src)
        .expect("something went wrong reading the file");

    let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();

    let program = Program::builder()
        .src(src)
        .cmplr_opt(kernel_half)
        .devices(device)
        .build(&context).unwrap();

    let sup_img_formats = Image::<u8>::supported_formats(&context, ocl::flags::MEM_READ_WRITE,
        MemObjectType::Image2d).unwrap();
    println!("Image formats supported: {}.", sup_img_formats.len());

    let dims = img.dimensions();

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

    let filter = Buffer::builder()
                .queue(queue.clone())
                .len(BUFF_SIZE * 4)
                .fill_val(BUFF_VAL)
                .build().unwrap();

    let kernel = Kernel::builder()
        .program(&program)
        .name("convolute")
        .queue(queue.clone())
        .gws(&dims)
        .arg_img(&src_image)
        .arg_img(&dst_image)
        .arg_buf(&filter)
        .build().unwrap();

    unsafe { kernel.enq().unwrap(); }

    dst_image.read(&mut img).enq().unwrap();

    // println!("{:?}",img);

    // let ref mut fout = File::create("files/out.jpg").unwrap();

    img.save(&Path::new(&OUTPUT_FILE)).unwrap();

    Ok(())
}



fn main() {
    // trivial().unwrap();
    // let img = image::open("files/img.jpg").unwrap();

    // let ref mut fout = File::create("files/out.jpg").unwrap();

    // img.save(fout,image::JPEG).unwrap();
    // paint_blue().unwrap();
    println!("O buffer está a ser passado com os valores a 0!!!!!");
    convolute().unwrap();
}