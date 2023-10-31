pub struct CoreObj<T> {
    pub some_item: Option<T>,
    pub some_numbers: [i32; 3],
}

pub trait CoreTrait<T> {
    fn get_some_number(&self, index: usize) -> i32;
    fn set_some_number(&mut self, some_number: i32, index: usize);
    
    fn get_some_item(&self) -> &Option<T>;
    fn set_some_item(&mut self, some_item: Option<T>);
}

impl <T> CoreObj<T> {
    pub fn new(some_numbers: [i32; 3], some_item: Option<T>) -> CoreObj<T> {
        CoreObj {
            some_numbers,
            some_item
        }
    }
}

impl <T> CoreTrait<T> for CoreObj<T> {
    fn get_some_number(&self, index: usize) -> i32 {
        self.some_numbers[index]
    }
    fn set_some_number(&mut self, some_number: i32, index: usize) {
        self.some_numbers[index] = some_number;
    }
    
    fn get_some_item(&self) -> &Option<T> {
        &self.some_item
    }
    fn set_some_item(&mut self, some_item: Option<T>) {
        self.some_item = some_item;
    }
}