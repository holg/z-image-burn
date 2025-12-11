/// Create a vector with evenly spaced floating point values.
///
/// This function generates a vector starting from `start`, ending at `end`, and incrementing by `step`.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (exclusive).
/// - `step`: An optional step value. If not provided, defaults to `1.0` if `start < end`, or `-1.0` if `start > end`.
///
/// # Returns
///
/// A vector containing the generated floating point values.
///
/// # Copyright
///
/// Shamelessly copied from:
/// https://github.com/crutcher/bimm/blob/main/crates/bimm/src/compat/ops.rs
/// Source is under MIT license.
#[must_use]
pub fn float_vec_arange(start: f64, end: f64, step: Option<f64>) -> Vec<f64> {
    assert_ne!(start, end);
    let step = if start < end {
        let step = step.unwrap_or(1.0);
        if step <= 0.0 {
            panic!("Step must be positive when start < end");
        }
        step
    } else {
        let step = step.unwrap_or(-1.0);
        if step >= 0.0 {
            panic!("Step must be negative when start > end");
        }
        step
    };

    let mut values: Vec<f64> = Vec::new();
    loop {
        let acc = start + values.len() as f64 * step;
        if (step > 0.0 && acc > end) || (step < 0.0 && acc < end) {
            break;
        }
        values.push(acc);
    }

    values
}

/// Create a vector with evenly spaced floating point values.
///
/// This function generates a vector with `num` values starting from `start`, ending at `end`, and evenly spaced.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (inclusive).
/// - `num`: The number of points to generate in the range.
///
/// # Returns
///
/// A vector containing the generated floating point values.
///
/// # Copyright
///
/// Shamelessly copied from:
/// https://github.com/crutcher/bimm/blob/main/crates/bimm/src/compat/ops.rs
/// Source is under MIT license.
#[must_use]
pub fn float_vec_linspace(start: f64, end: f64, num: usize) -> Vec<f64> {
    assert!(num > 0, "Number of points must be positive");

    if num == 1 {
        return vec![start];
    }

    let step = (end - start) / (num as f64 - 1.0);

    let end = if step > 0.0 {
        end + f64::EPSILON // Avoid floating point precision issues
    } else {
        end - f64::EPSILON // Avoid floating point precision issues
    };

    float_vec_arange(start, end, Some(step))
}
