extern crate ocl;
extern crate image;
#[macro_use] extern crate colorify;

use std::fs::File;
use std::io::Read;
use std::path::Path;
use ocl::ProQue;
use image::GenericImage;

use ocl::{Context, Queue, Device, Program, Image, Sampler, Kernel, Buffer, Event, EventList};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, AddressingMode, FilterMode, MemObjectType};
use ocl::flags::{CommandQueueProperties};
use ocl::enums::ProfilingInfo;




// fn trivial() -> ocl::Result<()> {
//     // let src = r#"
//     //     __kernel void add(__global float* buffer, float scalar) {
//     //         buffer[get_global_id(0)] += scalar;
//     //     }
//     // "#;

//     let mut f = File::open("src/kernels.cl").expect("file not found");

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

fn paint_blue() -> ocl::Result<()> {const KERNEL_SIZE: u32 = 31;
    const KERNEL_SIZE_HALF: u32 = KERNEL_SIZE / 2;
    const BUFF_SIZE: u32 = KERNEL_SIZE * KERNEL_SIZE;
    const BUFF_VAL: f32 = 1.0 / BUFF_SIZE as f32;
    let mut img = image::open("files/leninha.jpg")
                .unwrap()
                .to_rgba();
    let dims = img.dimensions();
    let row_S = dims.1;
    let col_S = dims.0;
    let kernel_half = format!("-D KERNEL_SIZE_HALF={}",KERNEL_SIZE_HALF);
    let row_size = format!("-D ROW_SIZE={}",row_S);
    let col_size = format!("-D COL_SIZE={}",col_S);
    let cl_opts = format!("{} {} {}",kernel_half, row_size,col_size);
    let index = 43;


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

    let sup_img_formats = Image::<u8>::supported_formats(&context, ocl::flags::MEM_READ_WRITE,
        MemObjectType::Image2d).unwrap();
    println!("Image formats supported: {}.", sup_img_formats.len());


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

    let row_S = dims.1;
    let col_S = dims.0;
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
    const KERNEL_SIZE: u32 = 3;
    const KERNEL_SIZE_HALF: u32 = KERNEL_SIZE / 2;
    const BUFF_SIZE: u32 = KERNEL_SIZE * KERNEL_SIZE;
    const BUFF_VAL: f32 = 1.0 / BUFF_SIZE as f32;
    let KERNELS: String = String::from("src/kernels.cl");
    let KERNEL_NAME: String = String::from("convolute");
    let FILE: String = String::from("leninha.jpg");
    let INPUT_FILE: String = format!("files/{}",FILE);
    let OUTPUT_FILE: String = format!("files/output_{}",FILE);
    let mut img = image::open(INPUT_FILE)
                .unwrap()
                .to_rgba();
    // Create a new ImgBuf with width: imgx and height: imgy
    let dims = img.dimensions();
    let row_S = dims.1;
    let col_S = dims.0;
    let mut res: image::ImageBuffer<image::Rgba<u8>, _> = image::ImageBuffer::new(row_S, col_S);
    let kernel_half = format!("-D KERNEL_SIZE_HALF={}",KERNEL_SIZE_HALF);
    let row_size = format!("-D ROW_SIZE={}",row_S);
    let col_size = format!("-D COL_SIZE={}",col_S);
    let cl_opts = format!("{} {} {}",kernel_half, row_size,col_size);
    


    let mut f = File::open(KERNELS).expect("file not found");

    let mut src = String::new();
    f.read_to_string(&mut src)
        .expect("something went wrong reading the file");

    println!("{}",src);

    let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
    let device = context.devices()[0];
    let queue = Queue::new(&context, device, Some(CommandQueueProperties::new().profiling())).unwrap();

    let program = Program::builder()
        .src(src)
        .cmplr_opt(cl_opts)
        .devices(device)
        .build(&context).unwrap();

    let sup_img_formats = Image::<u8>::supported_formats(&context, ocl::flags::MEM_READ_WRITE,
        MemObjectType::Image2d).unwrap();
    println!("Image formats supported: {}.", sup_img_formats.len());


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
        .copy_host_slice(&res)
        .queue(queue.clone())
        .build().unwrap();

    let filter = Buffer::builder()
                .queue(queue.clone())
                .len(BUFF_SIZE * 4)
                .fill_val(BUFF_VAL)
                .build().unwrap();

    let kernel = Kernel::builder()
        .program(&program)
        .name(KERNEL_NAME)
        .queue(queue.clone())
        .gws(&dims)
        .arg(&src_image)
        .arg(&dst_image)
        .arg(&filter)
        .build().unwrap();

    let mut event_time = EventList::new();

    unsafe { kernel.cmd().enew(& mut event_time).enq().unwrap(); }

    println!("Waiting");
    queue.finish().unwrap();
    println!("Finished");

    dst_image.read(&mut res).ewait(&event_time).enq().unwrap();

    let event = event_time.pop().unwrap();
    // let profiling = String::from(event_time.profiling_info());
    let start = event.profiling_info(ocl::enums::ProfilingInfo::Start).unwrap().time().unwrap();
    let end = event.profiling_info(ocl::enums::ProfilingInfo::End).unwrap().time().unwrap();
    let time = (end as f64 - start as f64) / 1000000000.0;

    println!("The kernel started at {}",start);
    println!("The kernel ended at {}",end);
    println!("The kernel took {} seconds to complete",time);

    // println!("{:?}",img);

    // let ref mut fout = File::create("files/out.jpg").unwrap();

    res.save(&Path::new(&OUTPUT_FILE)).unwrap();

    Ok(())
}



fn main() {
    // trivial().unwrap();
    // let img = image::open("files/img.jpg").unwrap();

    // let ref mut fout = File::create("files/out.jpg").unwrap();

    // img.save(fout,image::JPEG).unwrap();
    // paint_blue().unwrap();
    convolute().unwrap();
}