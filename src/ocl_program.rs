extern crate ocl;
extern crate image;

use crate::filter::{Square, Filter};

use std::fs::File;
use std::io::Read;
use std::path::Path;

use ocl::{Context, Queue, Device, Program, Image, Kernel, Buffer, EventList};
use ocl::enums::{ImageChannelOrder, ImageChannelDataType, MemObjectType};
use ocl::flags::{CommandQueueProperties};
use ocl::enums::ProfilingInfoResult;

pub struct OclProgram {
	input_image: Image<u8>,
	output_image: Image<u8>,
	output_file: String,
	kernel: Kernel,
	queue: Queue,
	start_profile: Option<ProfilingInfoResult>,
	end_profile: Option<ProfilingInfoResult>,
}

impl OclProgram {
	pub fn new(kernel_size: u32, kernel_file_name: String, kernel_name: String, image_name: String) -> OclProgram {
		let kernel_size_half: u32 = kernel_size / 2;
		let buff_size: u32 = kernel_size * kernel_size;
		let input_image: String = format!("files/{}",image_name);
		let output_file: String = format!("files/output_{}",image_name);
		let img = image::open(input_image)
					.unwrap()
					.to_rgba();
		// Create a new ImgBuf with width: imgx and height: imgy
		let dims = img.dimensions();
    	let res: image::ImageBuffer<image::Rgba<u8>, _> = image::ImageBuffer::new(dims.0, dims.1);
		let kernel_half = format!("-D KERNEL_SIZE_HALF={}",kernel_size_half);
		let row_size = format!("-D ROW_SIZE={}",dims.0);
		let col_size = format!("-D COL_SIZE={}",dims.1);
		let cl_opts = format!("{} {} {}",kernel_half, row_size,col_size);
		let mut f = File::open(kernel_file_name).expect("file not found");

		let mut src = String::new();
		f.read_to_string(&mut src)
			.expect("something went wrong reading the file");

		// println!("{}",src);

		let context = Context::builder().devices(Device::specifier().first()).build().unwrap();
		let device = context.devices()[0];
		let queue = Queue::new(&context, device, Some(CommandQueueProperties::new().profiling())).unwrap();

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
			.copy_host_slice(&res)
			.queue(queue.clone())
			.build().unwrap();


		let sq: Square = Square::new(&buff_size);

		// println!("{:?}", sq);

		let filter = Buffer::builder()
					.queue(queue.clone())
					.len(sq.size())
					.copy_host_slice(sq.as_slice())
					.build().unwrap();

		let kernel = Kernel::builder()
			.program(&program)
			.name(kernel_name)
			.queue(queue.clone())
			.global_work_size((dims.0,dims.1,1))
			.local_work_size((16, 16))
			.arg(&src_image)
			.arg(&dst_image)
			.arg(&filter)
			.build().unwrap();

		OclProgram {
			input_image: src_image,
			output_image: dst_image,
			output_file:output_file,
			kernel: kernel,
			queue: queue,
			start_profile: None,
			end_profile: None,
		}
	}


	pub fn run(&mut self) {
		let dims = self.input_image.dims().to_lens().unwrap();
		let mut res: image::ImageBuffer<image::Rgba<u8>, _> = image::ImageBuffer::new(dims[0] as u32, dims[1] as u32);
		let mut event_time = EventList::new();

		unsafe { self.kernel.cmd().enew(& mut event_time).enq().unwrap(); }

		println!("Waiting");
		self.queue.finish().unwrap();
		println!("Finished");

		self.output_image.read(&mut res).ewait(&event_time).enq().unwrap();

		let event = event_time.pop().unwrap();
		self.start_profile = Some(event.profiling_info(ocl::enums::ProfilingInfo::Start).unwrap());
		self.end_profile = Some(event.profiling_info(ocl::enums::ProfilingInfo::End).unwrap());

		res.save(&Path::new(&self.output_file)).unwrap();
	}

	pub fn print_profile(self) -> String {
		let start_profile = self.start_profile.unwrap();
		let start = start_profile.time().unwrap();
		let end = self.end_profile.unwrap().time().unwrap();

		let time = (end as f64 - start as f64) / 1_000_000_000.0;

		format!("The kernel started at {}, and ended at {}, taking {} seconds to complete",
				start,
				end,
				time)
	}
}
	
