extern crate ocl;
extern crate image;

mod filter;
mod ocl_program;

use ocl_program::OclProgram;

fn test() {
    let mut ocl = OclProgram::new(33, "src/kernels.cl".to_string(), "convolute_mem".to_string(), "leninha.jpg".to_string());
    // ocl.run();
    println!("{}", ocl.devices_info());
}

fn main() {

    test();

}