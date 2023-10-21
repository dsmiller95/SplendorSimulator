#![feature(impl_trait_in_assoc_type)]
#![feature(return_position_impl_trait_in_trait)]

pub mod game_model;
mod game_actions;
mod constants;

fn main() {
    println!("Hello, world!");

}

fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    unsafe {
        ::core::slice::from_raw_parts(
            (p as *const T) as *const u8,
            ::core::mem::size_of::<T>(),
        )
    }
}

/*
bytes: [1, 0, 0, 0, 1, 1, 0, 0, 104, 104, 104, 104, 104, 105, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 104, 104, 104, 104, 104, 105, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 104, 104, 104, 104, 104, 105, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 104, 104, 104, 104, 104, 105, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 104, 104, 104, 104, 104, 105, 0, 0, 1, 0
, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 1
07, 108, 0, 0, 0, 0, 0, 90, 0, 91, 0, 254, 255, 255, 255, 255, 255, 255, 255, 80, 233, 30, 136, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107
, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 0, 0, 0, 0, 90, 0, 91, 0, 254, 255, 255, 255, 255, 255, 255, 255, 80, 233, 30, 136, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106,
106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 1, 0, 0, 0, 0, 1, 0, 0, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 108, 0, 0, 0, 0, 0, 90, 0, 91, 0, 254, 255, 255, 255, 255, 255, 255, 255, 80, 233, 30, 136, 0, 0, 0, 0, 162,
 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 103, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0,
0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 103, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0
, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 103, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 162, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 101, 101, 101, 101, 101
, 102, 102, 102, 102, 102, 103, 100, 100, 100, 100, 100, 100, 245, 23]

 */


















