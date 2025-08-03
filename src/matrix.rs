//use std::fmt::Display;
use half::f16;

#[derive(Debug)]
pub struct Matrix<T> {
    pub data: Vec<T>,
    pub size: (usize, usize),
}

/*impl<T: Display> Matrix<T> {
    #[warn(dead_code)]
    pub fn data_size(&self) -> usize {
        self.size.0 * self.size.1 * size_of::<T>()
    }

    #[warn(dead_code)]
    pub fn print(&self) {
        for i in 0..self.size.0 {
            for j in 0..self.size.1 {
                print! {"{} ", self.data[i*self.size.1 + j]};
            }
            println!();
        }
        println!();
    }
}

impl Matrix<f32> {
    #[warn(dead_code)]
    pub fn from_bytes(bytes: &[u8], new_matrix_shape: &[usize]) -> Self {
        assert_eq!(new_matrix_shape.len(), 2);
        let new_matrix_size = (new_matrix_shape[0], new_matrix_shape[1]);
        let unit_size = size_of::<f32>();

        assert!(bytes.len() % unit_size == 0);

        let elements_nb = bytes.len() / unit_size;

        assert_eq!(elements_nb, new_matrix_size.0 * new_matrix_size.1);

        let mut mat = Self {
            data: Vec::with_capacity(elements_nb),
            size: new_matrix_size,
        };

        for chunk in bytes.chunks(unit_size) {
            let float_bytes = chunk.try_into().unwrap();
            let float_val = f32::from_le_bytes(float_bytes);

            mat.data.push(float_val);
        }

        mat
    }
}*/

impl Matrix<f16> {
    pub fn from_bytes(bytes: &[u8], new_matrix_shape: &[usize]) -> Self {
        assert_eq!(new_matrix_shape.len(), 2);
        let new_matrix_size = (new_matrix_shape[0], new_matrix_shape[1]);
        let unit_size = size_of::<f16>();

        assert!(bytes.len() % unit_size == 0);

        let elements_nb = bytes.len() / unit_size;

        assert_eq!(elements_nb, new_matrix_size.0 * new_matrix_size.1);

        let mut mat = Self {
            data: Vec::with_capacity(elements_nb),
            size: new_matrix_size,
        };

        for chunk in bytes.chunks_exact(unit_size) {
            let bytes_array: [u8; 2] = chunk.try_into().unwrap();
            let float_val = f16::from_le_bytes(bytes_array);

            mat.data.push(float_val);
        }

        mat
    }
}
