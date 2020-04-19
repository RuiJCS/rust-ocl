

// #[derive(Debug)]
// enum FilterType {
//     SQUARE,
//     CIRCLE,
//     GAUSSIAN,
// }

#[derive(Debug)]
pub struct Box {
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


impl Box {
    pub fn new (fsize:&u32) -> Self {
        let size: u32 = fsize * 4;
        let value: f32 = 1.0 / *fsize as f32;
        let values = vec![value; size as usize];
        Self {size, values}
    }
}

impl Filter for Box {
    fn as_slice(&self) -> &[f32] {
        self.values.as_slice()
    }

    fn size(&self) -> u32 {
        // println!("{}",self.size);
        self.size
    }
}

#[derive(Debug)]
pub struct Edge {
    values: Vec<f32>,
    size: u32,
}

impl Edge {
    pub fn new (fsize: &u32) -> Self {
        let size: u32 = fsize * 4;
        let mut values = vec![-1.; size as usize];
        values[15] = 8.;
        values[16] = 8.;
        values[17] = 8.;
        values[18] = 8.;
        Self {size, values}
    }
}

impl Filter for Edge {
    fn as_slice(&self) -> &[f32] {
        self.values.as_slice()
    }

    fn size(&self) -> u32 {
        // println!("{}",self.size);
        self.size
    }
}