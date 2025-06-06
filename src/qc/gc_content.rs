use std::sync::Arc;

use arrow_array::{Array, StringArray, ArrayRef};
use arrow_array::builder::UInt8Builder;
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{create_udf, ColumnarValue, Volatility};

pub fn register(ctx: &SessionContext) {
    let func = |args: &[ColumnarValue]| -> Result<ColumnarValue> {
        let seqs = match &args[0] {
            ColumnarValue::Array(arr) => arr
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal("gc_percent: expected Utf8 array".into())
                })?,
            _ => {
                return Err(DataFusionError::Internal(
                    "gc_percent: expected array argument".into(),
                ))
            }
        };

        let mut builder = UInt8Builder::with_capacity(seqs.len());

        for i in 0..seqs.len() {
            if seqs.is_null(i) {
                builder.append_null();
                continue;
            }
            let s = seqs.value(i).as_bytes();
            let gc = s
                .iter()
                .filter(|b| matches!(b, b'G' | b'g' | b'C' | b'c'))
                .count();
            builder.append_value(((gc * 100) / s.len()) as u8);
        }

        Ok(ColumnarValue::Array(Arc::new(builder.finish()) as ArrayRef))
    };

    let udf = create_udf(
        "gc_percent",
        vec![DataType::Utf8],
        DataType::UInt8,
        Volatility::Immutable,
        Arc::new(func),
    );
    ctx.register_udf(udf);
}
