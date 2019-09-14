

// #[derive(Debug)]
// enum FilterType {
//     SQUARE,
//     CIRCLE,
//     GAUSSIAN,
// }

#[derive(Debug)]
pub struct Square {
    values: Vec<f32>,
    size: u32,
    // type: FilterType,
}

pub trait Filter {
    fn as_slice(&self) -> &[f32] {
        // unimplemented!();
        panic!("Not implemented yet!!!!");
    }

    fn size(&self) -> u32 {
        panic!("Not implemented yet!!!!");
    }
}


impl Square {
    pub fn new (fsize:&u32) -> Self {
        let size: u32 = fsize * 4;
        let value: f32 = 1.0 / *fsize as f32;
        let values = vec![value; size as usize];
        Self {size, values}
    }
}

impl Filter for Square {
    fn as_slice(&self) -> &[f32] {
        self.values.as_slice()
    }

    fn size(&self) -> u32 {
        println!("{}",self.size);
        self.size
    }
}