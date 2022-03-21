pub mod som;
pub mod ndarray;

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);

    }

    // fn test_uniform() {
    //     let a = uniform((2, 3));
    //     let b = array![
    //         [0.0, 0.0],
    //         [0.0, 1.0],
    //         [1.0, 0.0],
    //         [1.0, 1.0],
    //         [2.0, 0.0],
    //         [2.0, 1.0]
    //     ];

    //     assert_eq!(a, b);
    // }
}
