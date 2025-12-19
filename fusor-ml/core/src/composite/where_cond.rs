use crate::{
    DataType, Tensor,
    compute_graph::NodeIndex,
    tensor::DataTypeEnum,
};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn where_cond<D2>(self, on_true: &Tensor<R, D2>, on_false: &Tensor<R, D2>) -> Tensor<R, D2>
    where
        D2: DataType,
    {
        let operation = WhereCondOperation::new(
            self.key(),
            on_true.key(),
            on_false.key(),
            self.datatype(),
            on_true.datatype(),
            self.shape(),
        );
        let data = on_true.data();

        Tensor::from_parts(data.where_cond(operation))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct WhereCondOperation {
    pub(crate) condition: NodeIndex,
    pub(crate) on_true: NodeIndex,
    pub(crate) on_false: NodeIndex,
    pub(crate) condition_datatype: DataTypeEnum,
    pub(crate) output_datatype: DataTypeEnum,
    pub(crate) shape: Box<[usize]>,
}

impl WhereCondOperation {
    pub fn new(
        condition: NodeIndex,
        on_true: NodeIndex,
        on_false: NodeIndex,
        condition_datatype: DataTypeEnum,
        output_datatype: DataTypeEnum,
        shape: &[usize],
    ) -> Self {
        Self {
            condition,
            on_true,
            on_false,
            condition_datatype,
            output_datatype,
            shape: shape.into(),
        }
    }

    pub(crate) fn to_nary(&self) -> crate::nary_wise::NaryOperation {
        use crate::nary_wise::NaryExpr;

        crate::nary_wise::NaryOperation {
            inputs: vec![self.condition, self.on_true, self.on_false],
            expression: NaryExpr::select(
                NaryExpr::input(0, self.condition_datatype),
                NaryExpr::input(1, self.output_datatype),
                NaryExpr::input(2, self.output_datatype),
                self.condition_datatype,
                self.output_datatype,
            ),
            shape: self.shape.clone(),
            output_datatype: self.output_datatype,
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_where_cond() {
    use crate::Device;

    let device = Device::test_instance();

    let data = Tensor::arange(&device, 0., 10.);
    let even = Tensor::arange(&device, 0, 10) % 2;
    let zero = Tensor::splat(&device, 0., *data.shape());

    let data_where_even = even.where_cond(&data, &zero);

    let result = data_where_even.as_slice().await.unwrap();
    println!("result: {result:?}");

    assert_eq!(result[[0]], 0.);
    assert_eq!(result[[1]], 1.);
    assert_eq!(result[[2]], 0.);
    assert_eq!(result[[3]], 3.);
    assert_eq!(result[[4]], 0.);
    assert_eq!(result[[5]], 5.);
    assert_eq!(result[[6]], 0.);
    assert_eq!(result[[7]], 7.);
    assert_eq!(result[[8]], 0.);
    assert_eq!(result[[9]], 9.);
}
