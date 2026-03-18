use std::{
    any::Any,
    collections::{HashMap, HashSet, VecDeque},
    ops::Range,
    sync::{Arc, Mutex},
};

use crate::{Device, Error, Result, Tensor as RawTensor, ToVec1, ToVec2, layers::Embedding};

type NodeId = usize;
type BackwardRule =
    Arc<dyn Fn(Box<dyn AnyTensorValue>) -> Result<Vec<BackwardTarget>> + Send + Sync>;

#[derive(Clone)]
pub struct Graph {
    inner: Arc<GraphInner>,
}

#[derive(Clone)]
pub struct Tensor<const R: usize> {
    value: RawTensor<R, f32>,
    handle: NodeHandle,
}

pub struct Gradients {
    gradients: HashMap<NodeId, Box<dyn AnyTensorValue>>,
}

pub struct BackwardTarget {
    node: NodeId,
    gradient: Box<dyn AnyTensorValue>,
}

#[derive(Clone)]
pub struct Parent {
    handle: NodeHandle,
}

#[derive(Clone)]
struct NodeHandle {
    graph: Arc<GraphInner>,
    id: NodeId,
}

#[derive(Clone)]
struct Node {
    parents: Vec<NodeId>,
    backward: Option<BackwardRule>,
    requires_grad: bool,
}

struct GraphInner {
    state: Mutex<GraphState>,
}

struct GraphState {
    next_id: NodeId,
    nodes: HashMap<NodeId, Node>,
}

trait AnyTensorValue: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn AnyTensorValue>;
    fn add_box(&self, other: &dyn AnyTensorValue) -> Result<Box<dyn AnyTensorValue>>;
}

impl Graph {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(GraphInner {
                state: Mutex::new(GraphState {
                    next_id: 0,
                    nodes: HashMap::new(),
                }),
            }),
        }
    }

    pub fn leaf<const R: usize>(&self, value: RawTensor<R, f32>) -> Tensor<R> {
        self.tensor_with_grad(value, true)
    }

    pub fn constant<const R: usize>(&self, value: RawTensor<R, f32>) -> Tensor<R> {
        self.tensor_with_grad(value, false)
    }

    pub fn tensor<const R: usize, T>(&self, device: &Device, data: T) -> Tensor<R>
    where
        RawTensor<R, f32>: fusor_types::FromArray<R, f32, T, Device>,
    {
        self.leaf(RawTensor::new(device, data))
    }

    pub fn constant_from_data<const R: usize, T>(&self, device: &Device, data: T) -> Tensor<R>
    where
        RawTensor<R, f32>: fusor_types::FromArray<R, f32, T, Device>,
    {
        self.constant(RawTensor::new(device, data))
    }

    fn tensor_with_grad<const R: usize>(
        &self,
        value: RawTensor<R, f32>,
        requires_grad: bool,
    ) -> Tensor<R> {
        let id = self.inner.add_node(Vec::new(), None, requires_grad);
        Tensor {
            value,
            handle: NodeHandle {
                graph: self.inner.clone(),
                id,
            },
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl<const R: usize> Tensor<R> {
    pub fn from_raw(graph: &Graph, value: RawTensor<R, f32>) -> Self {
        graph.leaf(value)
    }

    pub fn constant_from_raw(graph: &Graph, value: RawTensor<R, f32>) -> Self {
        graph.constant(value)
    }

    pub fn new<T>(graph: &Graph, device: &Device, data: T) -> Self
    where
        RawTensor<R, f32>: fusor_types::FromArray<R, f32, T, Device>,
    {
        graph.tensor(device, data)
    }

    pub fn from_slice(graph: &Graph, device: &Device, shape: [usize; R], data: &[f32]) -> Self {
        graph.leaf(RawTensor::from_slice(device, shape, data))
    }

    pub fn zeros(graph: &Graph, device: &Device, shape: [usize; R]) -> Self {
        graph.leaf(RawTensor::zeros(device, shape))
    }

    pub fn splat(graph: &Graph, device: &Device, value: f32, shape: [usize; R]) -> Self {
        graph.leaf(RawTensor::splat(device, value, shape))
    }

    pub fn raw(&self) -> &RawTensor<R, f32> {
        &self.value
    }

    pub fn into_raw(self) -> RawTensor<R, f32> {
        self.value
    }

    pub fn shape(&self) -> [usize; R] {
        self.value.shape()
    }

    pub fn device(&self) -> Device {
        self.value.device()
    }

    pub fn graph(&self) -> Graph {
        Graph {
            inner: self.handle.graph.clone(),
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.handle.graph.requires_grad(self.handle.id)
    }

    pub fn parent(&self) -> Parent {
        Parent {
            handle: self.handle.clone(),
        }
    }

    pub fn detach(&self) -> Self {
        let requires_grad = self.requires_grad();
        let id = self.handle.graph.add_node(Vec::new(), None, requires_grad);
        Self {
            value: self.value.to_concrete(),
            handle: NodeHandle {
                graph: self.handle.graph.clone(),
                id,
            },
        }
    }

    pub fn with_backwards<I, F>(self, parents: I, backwards: F) -> Self
    where
        I: IntoIterator<Item = Parent>,
        F: Fn(RawTensor<R, f32>) -> Result<Vec<BackwardTarget>> + Send + Sync + 'static,
    {
        let parent_handles = parents
            .into_iter()
            .map(|parent| parent.handle)
            .collect::<Vec<_>>();
        let requires_grad = parent_handles
            .iter()
            .any(|parent| parent.graph.requires_grad(parent.id));
        let parent_ids = parent_handles
            .iter()
            .map(|parent| parent.id)
            .collect::<Vec<_>>();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = gradient
                .as_any()
                .downcast_ref::<RawTensor<R, f32>>()
                .ok_or_else(|| Error::msg("gradient rank mismatch in custom backward"))?
                .clone();
            backwards(gradient)
        });
        self.handle.graph.replace_node(
            self.handle.id,
            Node {
                parents: parent_ids,
                backward: Some(backward),
                requires_grad,
            },
        );
        self
    }

    pub fn backward(&self) -> Result<Gradients> {
        let elements = self.shape().iter().product::<usize>();
        if elements != 1 {
            return Err(Error::msg(
                "backward() requires a single-element tensor; use backward_with() for non-scalars",
            ));
        }
        let seed = RawTensor::splat(&self.device(), 1.0, self.shape());
        self.backward_with(seed)
    }

    pub fn backward_with(&self, seed: RawTensor<R, f32>) -> Result<Gradients> {
        self.handle.graph.backward(self.handle.id, Box::new(seed))
    }

    fn from_op<const OUT: usize>(
        &self,
        value: RawTensor<OUT, f32>,
        parents: Vec<NodeHandle>,
        backward: Option<BackwardRule>,
    ) -> Tensor<OUT> {
        for parent in &parents {
            assert!(
                Arc::ptr_eq(&self.handle.graph, &parent.graph),
                "cannot mix autograd tensors from different graphs"
            );
        }
        let requires_grad = parents
            .iter()
            .any(|parent| parent.graph.requires_grad(parent.id));
        let parent_ids = parents.into_iter().map(|parent| parent.id).collect();
        let id = self
            .handle
            .graph
            .add_node(parent_ids, backward, requires_grad);
        Tensor {
            value,
            handle: NodeHandle {
                graph: self.handle.graph.clone(),
                id,
            },
        }
    }

    pub fn add(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() + rhs.value.clone()).to_concrete(),
            |grad, _, _| vec![grad.clone().to_concrete(), grad.to_concrete()],
        )
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() - rhs.value.clone()).to_concrete(),
            |grad, _, _| vec![grad.clone().to_concrete(), (-grad).to_concrete()],
        )
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() * rhs.value.clone()).to_concrete(),
            |grad, lhs, rhs| {
                vec![
                    (grad.clone() * rhs).to_concrete(),
                    (grad * lhs).to_concrete(),
                ]
            },
        )
    }

    pub fn div(&self, rhs: &Self) -> Self {
        self.binary_op(
            rhs,
            (self.value.clone() / rhs.value.clone()).to_concrete(),
            |grad, lhs, rhs| {
                let lhs_grad = (grad.clone() / rhs.clone()).to_concrete();
                let rhs_grad = (-((grad * lhs) / rhs.sqr().to_concrete())).to_concrete();
                vec![lhs_grad, rhs_grad]
            },
        )
    }

    pub fn add_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(self.value.add_scalar(scalar), move |grad, _| grad)
    }

    pub fn sub_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(self.value.sub_scalar(scalar), move |grad, _| grad)
    }

    pub fn mul_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.mul_scalar(scalar).to_concrete(),
            move |grad, _| grad.mul_scalar(scalar).to_concrete(),
        )
    }

    pub fn div_scalar(&self, scalar: f32) -> Self {
        self.unary_from_value(
            self.value.div_scalar(scalar).to_concrete(),
            move |grad, _| grad.div_scalar(scalar).to_concrete(),
        )
    }

    pub fn neg(&self) -> Self {
        self.unary_from_value((-self.value.clone()).to_concrete(), move |grad, _| {
            (-grad).to_concrete()
        })
    }

    pub fn sqr(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.sqr().to_concrete(), move |grad, _| {
            ((grad * input.clone()).to_concrete().mul_scalar(2.0)).to_concrete()
        })
    }

    pub fn relu(&self) -> Self {
        let output = self.value.relu().to_concrete();
        self.unary_from_value(output.clone(), move |grad, out| {
            let zeros = RawTensor::zeros(&out.device(), out.shape());
            let ones = RawTensor::splat(&out.device(), 1.0, out.shape());
            (grad * out.where_cond(&ones, &zeros)).to_concrete()
        })
    }

    pub fn tanh(&self) -> Self {
        self.unary_from_value(self.value.tanh().to_concrete(), move |grad, out| {
            let one_minus_sq = (RawTensor::splat(&out.device(), 1.0, out.shape())
                - out.sqr().to_concrete())
            .to_concrete();
            (grad * one_minus_sq).to_concrete()
        })
    }

    pub fn exp(&self) -> Self {
        self.unary_from_value(self.value.exp().to_concrete(), move |grad, out| {
            (grad * out).to_concrete()
        })
    }

    pub fn log(&self) -> Self {
        let input = self.value.clone();
        self.unary_from_value(self.value.log().to_concrete(), move |grad, _| {
            (grad / input.clone()).to_concrete()
        })
    }

    pub fn sqrt(&self) -> Self {
        self.unary_from_value(self.value.sqrt().to_concrete(), move |grad, out| {
            let denom = out.mul_scalar(2.0).to_concrete();
            (grad / denom).to_concrete()
        })
    }

    pub fn reshape<const OUT: usize>(&self, shape: [usize; OUT]) -> Tensor<OUT> {
        let input_shape = self.shape();
        let value = self.value.reshape(shape).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "reshape")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.reshape(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let value = self.value.transpose(dim0, dim1).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "transpose")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.transpose(dim0, dim1).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn slice(&self, slices: [Range<usize>; R]) -> Self {
        let input_shape = self.shape();
        let value = self.value.slice(slices.clone()).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "slice")?;
            let zeros = RawTensor::zeros(&gradient.device(), input_shape);
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(zeros.slice_assign(slices.clone(), &gradient).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn broadcast_as<const OUT: usize>(&self, shape: [usize; OUT]) -> Tensor<OUT> {
        let input_shape = self.shape();
        let value = self.value.broadcast_as(shape).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<OUT>(&*gradient, "broadcast_as")?;
            let reduced = reduce_broadcast_gradient(gradient, input_shape)?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: reduced,
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn unary_from_value(
        &self,
        value: RawTensor<R, f32>,
        backward: impl Fn(RawTensor<R, f32>, RawTensor<R, f32>) -> RawTensor<R, f32>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        let input_id = self.handle.id;
        let output = value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "unary")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(backward(gradient, output.clone()).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    fn binary_op(
        &self,
        rhs: &Self,
        value: RawTensor<R, f32>,
        backward: impl Fn(
            RawTensor<R, f32>,
            RawTensor<R, f32>,
            RawTensor<R, f32>,
        ) -> Vec<RawTensor<R, f32>>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        assert!(
            Arc::ptr_eq(&self.handle.graph, &rhs.handle.graph),
            "cannot mix autograd tensors from different graphs"
        );
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<R>(&*gradient, "binary")?;
            let gradients = backward(gradient, lhs_value.clone(), rhs_value.clone());
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(gradients[0].clone().to_concrete()),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(gradients[1].clone().to_concrete()),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }
}

impl Tensor<1> {
    pub fn sum(&self) -> Tensor<0> {
        let input_shape = self.shape();
        let value = self.value.sum::<0>(0);
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<0>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor<2> {
        let value = self.value.unsqueeze(dim).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "unsqueeze")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.squeeze(dim).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }
}

impl Tensor<2> {
    pub fn mat_mul(&self, rhs: &Tensor<2>) -> Tensor<2> {
        assert_same_graph(self, rhs);
        let value = self.value.mat_mul(&rhs.value);
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "mat_mul")?;
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(gradient.clone().mat_mul(&rhs_value.transpose(0, 1))),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(lhs_value.transpose(0, 1).mat_mul(&gradient)),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    pub fn squeeze(&self, dim: usize) -> Tensor<1> {
        let value = self.value.squeeze(dim).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<1>(&*gradient, "squeeze")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.unsqueeze(dim).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn unsqueeze(&self, dim: usize) -> Tensor<3> {
        let value = self.value.unsqueeze(dim).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "unsqueeze")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.squeeze(dim).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum(&self, axis: usize) -> Tensor<1> {
        let input_shape = self.shape();
        let value = self.value.sum::<1>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<1>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(
                    gradient
                        .unsqueeze(axis)
                        .broadcast_as(input_shape)
                        .to_concrete(),
                ),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum_keepdim(&self, axis: usize) -> Tensor<2> {
        let input_shape = self.shape();
        let value = self.value.sum_keepdim::<1>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "sum_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn gather_last(&self, indices: &RawTensor<1, u32>) -> Tensor<1> {
        let shape = self.shape();
        assert_eq!(
            shape[0],
            indices.shape()[0],
            "gather_last expects one index per row"
        );
        let width = shape[1];
        let device = self.device();
        let index_values = pollster::block_on(indices.clone().as_slice())
            .unwrap()
            .to_vec1();
        let linear_indices = index_values
            .iter()
            .enumerate()
            .map(|(row, &column)| {
                assert!(
                    (column as usize) < width,
                    "gather_last index {} out of bounds for width {}",
                    column,
                    width
                );
                (row * width + column as usize) as u32
            })
            .collect::<Vec<_>>();
        let linear_indices_tensor = RawTensor::from_slice(&device, [shape[0]], &linear_indices);
        let flat = self.value.reshape([shape[0] * width]).to_concrete();
        let value = flat.index_select(0, &linear_indices_tensor).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<1>(&*gradient, "gather_last")?;
            let gradient_values = pollster::block_on(gradient.clone().as_slice())?.to_vec1();
            let mut input_gradient = vec![0.0f32; shape[0] * width];
            for (row, &linear_index) in linear_indices.iter().enumerate() {
                input_gradient[linear_index as usize] += gradient_values[row];
            }
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(RawTensor::from_slice(&device, shape, &input_gradient)),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn embedding(&self, indices: &RawTensor<2, u32>) -> Tensor<3> {
        let value: RawTensor<3, f32> =
            Embedding::new_from_tensor(self.value.clone()).forward(indices);
        let table_id = self.handle.id;
        let table_shape = self.shape();
        let device = self.device();
        let indices = indices.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "embedding")?;
            let index_values = pollster::block_on(indices.clone().as_slice())?.to_vec2();
            let grad_shape = gradient.shape();
            let grad_flat = gradient.reshape([grad_shape[0] * grad_shape[1], grad_shape[2]]);

            let mut rows_by_token = HashMap::<u32, Vec<u32>>::new();
            for (batch, row) in index_values.iter().enumerate() {
                for (position, &token) in row.iter().enumerate() {
                    let flat_row = (batch * grad_shape[1] + position) as u32;
                    rows_by_token.entry(token).or_default().push(flat_row);
                }
            }

            let mut embedding_gradient = RawTensor::zeros(&device, table_shape);
            for (token, rows) in rows_by_token {
                let row_indices = RawTensor::from_slice(&device, [rows.len()], &rows);
                let token_gradient = grad_flat
                    .index_select(0, &row_indices)
                    .sum::<1>(0)
                    .unsqueeze::<2>(0)
                    .to_concrete();
                embedding_gradient = embedding_gradient.slice_assign(
                    [token as usize..token as usize + 1, 0..table_shape[1]],
                    &token_gradient,
                );
            }

            Ok(vec![BackwardTarget {
                node: table_id,
                gradient: Box::new(embedding_gradient),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }
}

impl Tensor<3> {
    pub fn mat_mul(&self, rhs: &Tensor<3>) -> Tensor<3> {
        assert_same_graph(self, rhs);
        let value = self.value.mat_mul(&rhs.value);
        let lhs_id = self.handle.id;
        let rhs_id = rhs.handle.id;
        let lhs_value = self.value.clone();
        let rhs_value = rhs.value.clone();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "mat_mul")?;
            Ok(vec![
                BackwardTarget {
                    node: lhs_id,
                    gradient: Box::new(gradient.clone().mat_mul(&rhs_value.transpose(1, 2))),
                },
                BackwardTarget {
                    node: rhs_id,
                    gradient: Box::new(lhs_value.transpose(1, 2).mat_mul(&gradient)),
                },
            ])
        });
        self.from_op(
            value,
            vec![self.handle.clone(), rhs.handle.clone()],
            Some(backward),
        )
    }

    pub fn squeeze(&self, dim: usize) -> Tensor<2> {
        let value = self.value.squeeze(dim).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "squeeze")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.unsqueeze(dim).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum(&self, axis: usize) -> Tensor<2> {
        let input_shape = self.shape();
        let value = self.value.sum::<2>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<2>(&*gradient, "sum")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(
                    gradient
                        .unsqueeze(axis)
                        .broadcast_as(input_shape)
                        .to_concrete(),
                ),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn sum_keepdim(&self, axis: usize) -> Tensor<3> {
        let input_shape = self.shape();
        let value = self.value.sum_keepdim::<2>(axis).to_concrete();
        let input_id = self.handle.id;
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "sum_keepdim")?;
            Ok(vec![BackwardTarget {
                node: input_id,
                gradient: Box::new(gradient.broadcast_as(input_shape).to_concrete()),
            }])
        });
        self.from_op(value, vec![self.handle.clone()], Some(backward))
    }

    pub fn cat(tensors: Vec<Tensor<3>>, dim: usize) -> Tensor<3> {
        assert!(!tensors.is_empty(), "cat requires at least one tensor");
        let graph = tensors[0].handle.graph.clone();
        let raw = tensors
            .iter()
            .map(|tensor| tensor.value.clone())
            .collect::<Vec<_>>();
        let value = RawTensor::cat(raw, dim);
        let parents = tensors
            .iter()
            .map(|tensor| tensor.handle.clone())
            .collect::<Vec<_>>();
        let parent_ids = parents.iter().map(|parent| parent.id).collect::<Vec<_>>();
        let slices = tensors
            .iter()
            .scan(0usize, |offset, tensor| {
                let start = *offset;
                let length = tensor.shape()[dim];
                *offset += length;
                Some(start..start + length)
            })
            .collect::<Vec<_>>();
        let backward: BackwardRule = Arc::new(move |gradient| {
            let gradient = downcast_tensor::<3>(&*gradient, "cat")?;
            let mut targets = Vec::with_capacity(parent_ids.len());
            for (&parent_id, slice) in parent_ids.iter().zip(slices.iter()) {
                let grad_slice = match dim {
                    0 => gradient.slice([
                        slice.clone(),
                        0..gradient.shape()[1],
                        0..gradient.shape()[2],
                    ]),
                    1 => gradient.slice([
                        0..gradient.shape()[0],
                        slice.clone(),
                        0..gradient.shape()[2],
                    ]),
                    2 => gradient.slice([
                        0..gradient.shape()[0],
                        0..gradient.shape()[1],
                        slice.clone(),
                    ]),
                    _ => panic!("invalid cat dim"),
                }
                .to_concrete();
                targets.push(BackwardTarget {
                    node: parent_id,
                    gradient: Box::new(grad_slice),
                });
            }
            Ok(targets)
        });
        let id = graph.add_node(
            parents.iter().map(|parent| parent.id).collect(),
            Some(backward),
            parents
                .iter()
                .any(|parent| parent.graph.requires_grad(parent.id)),
        );
        Tensor {
            value,
            handle: NodeHandle { graph, id },
        }
    }

    pub fn layer_norm(&self, weight: &Tensor<1>, bias: Option<&Tensor<1>>, eps: f32) -> Tensor<3> {
        let centered = {
            let mean = self.sum_keepdim(2).div_scalar(self.shape()[2] as f32);
            self.sub(&mean.broadcast_as(self.shape()))
        };
        let variance = centered
            .sqr()
            .sum_keepdim(2)
            .div_scalar(self.shape()[2] as f32);
        let std = variance.add_scalar(eps).sqrt();
        let normalized = centered.div(&std.broadcast_as(self.shape()));
        let scaled = normalized.mul(&weight.broadcast_as(self.shape()));
        if let Some(bias) = bias {
            scaled.add(&bias.broadcast_as(self.shape()))
        } else {
            scaled
        }
    }
}

impl Gradients {
    pub fn get<const R: usize>(&self, tensor: &Tensor<R>) -> Option<RawTensor<R, f32>> {
        self.gradients
            .get(&tensor.handle.id)
            .and_then(|gradient| gradient.as_any().downcast_ref::<RawTensor<R, f32>>())
            .cloned()
    }
}

impl BackwardTarget {
    pub fn wrt<const R: usize>(tensor: &Tensor<R>, gradient: RawTensor<R, f32>) -> Self {
        Self {
            node: tensor.handle.id,
            gradient: Box::new(gradient),
        }
    }
}

impl GraphInner {
    fn add_node(
        &self,
        parents: Vec<NodeId>,
        backward: Option<BackwardRule>,
        requires_grad: bool,
    ) -> NodeId {
        let mut state = self.state.lock().unwrap();
        let id = state.next_id;
        state.next_id += 1;
        state.nodes.insert(
            id,
            Node {
                parents,
                backward,
                requires_grad,
            },
        );
        id
    }

    fn replace_node(&self, id: NodeId, node: Node) {
        self.state.lock().unwrap().nodes.insert(id, node);
    }

    fn requires_grad(&self, id: NodeId) -> bool {
        self.state
            .lock()
            .unwrap()
            .nodes
            .get(&id)
            .map(|node| node.requires_grad)
            .unwrap_or(false)
    }

    fn backward(&self, root: NodeId, seed: Box<dyn AnyTensorValue>) -> Result<Gradients> {
        let nodes = self.reachable_nodes(root);
        let mut pending_children = HashMap::<NodeId, usize>::new();
        for (id, node) in &nodes {
            pending_children.entry(*id).or_insert(0);
            for parent in &node.parents {
                *pending_children.entry(*parent).or_insert(0) += 1;
            }
        }

        let mut gradients = HashMap::<NodeId, Box<dyn AnyTensorValue>>::new();
        gradients.insert(root, seed);

        let mut queue = VecDeque::new();
        queue.push_back(root);

        while let Some(node_id) = queue.pop_front() {
            let Some(node) = nodes.get(&node_id) else {
                continue;
            };
            let Some(backward) = node.backward.as_ref() else {
                continue;
            };
            let gradient = gradients
                .get(&node_id)
                .ok_or_else(|| Error::msg(format!("missing gradient for node {node_id}")))?
                .clone_box();

            for target in backward(gradient)? {
                let Some(parent_node) = nodes.get(&target.node) else {
                    continue;
                };
                if !parent_node.requires_grad {
                    continue;
                }
                accumulate_gradient(&mut gradients, target.node, target.gradient)?;
                let remaining = pending_children.get_mut(&target.node).ok_or_else(|| {
                    Error::msg(format!("missing child count for node {}", target.node))
                })?;
                *remaining = remaining.saturating_sub(1);
                if *remaining == 0 {
                    queue.push_back(target.node);
                }
            }
        }

        Ok(Gradients { gradients })
    }

    fn reachable_nodes(&self, root: NodeId) -> HashMap<NodeId, Node> {
        let snapshot = self.state.lock().unwrap().nodes.clone();
        let mut reachable = HashMap::new();
        let mut stack = vec![root];
        let mut visited = HashSet::new();
        while let Some(node_id) = stack.pop() {
            if !visited.insert(node_id) {
                continue;
            }
            if let Some(node) = snapshot.get(&node_id) {
                reachable.insert(node_id, node.clone());
                stack.extend(node.parents.iter().copied());
            }
        }
        reachable
    }
}

impl<const R: usize> AnyTensorValue for RawTensor<R, f32> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn AnyTensorValue> {
        Box::new(self.clone())
    }

    fn add_box(&self, other: &dyn AnyTensorValue) -> Result<Box<dyn AnyTensorValue>> {
        let other = other
            .as_any()
            .downcast_ref::<RawTensor<R, f32>>()
            .ok_or_else(|| Error::msg("gradient rank mismatch while accumulating"))?;
        Ok(Box::new((self.clone() + other.clone()).to_concrete()))
    }
}

fn accumulate_gradient(
    gradients: &mut HashMap<NodeId, Box<dyn AnyTensorValue>>,
    node: NodeId,
    gradient: Box<dyn AnyTensorValue>,
) -> Result<()> {
    match gradients.get(&node) {
        Some(existing) => {
            let accumulated = existing.add_box(&*gradient)?;
            gradients.insert(node, accumulated);
        }
        None => {
            gradients.insert(node, gradient);
        }
    }
    Ok(())
}

fn downcast_tensor<const R: usize>(
    value: &dyn AnyTensorValue,
    context: &str,
) -> Result<RawTensor<R, f32>> {
    value
        .as_any()
        .downcast_ref::<RawTensor<R, f32>>()
        .cloned()
        .ok_or_else(|| Error::msg(format!("gradient rank mismatch in {context}")))
}

fn assert_same_graph<const R: usize, const R2: usize>(lhs: &Tensor<R>, rhs: &Tensor<R2>) {
    assert!(
        Arc::ptr_eq(&lhs.handle.graph, &rhs.handle.graph),
        "cannot mix autograd tensors from different graphs"
    );
}

fn reduce_broadcast_gradient<const IN: usize, const OUT: usize>(
    gradient: RawTensor<OUT, f32>,
    input_shape: [usize; IN],
) -> Result<Box<dyn AnyTensorValue>> {
    match (IN, OUT) {
        (1, 1) => Ok(Box::new(reduce_same_rank_broadcast_1(
            gradient.reshape([gradient.shape()[0]]).to_concrete(),
            [input_shape[0]],
        ))),
        (2, 2) => Ok(Box::new(reduce_same_rank_broadcast_2(
            gradient
                .reshape([gradient.shape()[0], gradient.shape()[1]])
                .to_concrete(),
            [input_shape[0], input_shape[1]],
        ))),
        (3, 3) => Ok(Box::new(reduce_same_rank_broadcast_3(
            gradient
                .reshape([
                    gradient.shape()[0],
                    gradient.shape()[1],
                    gradient.shape()[2],
                ])
                .to_concrete(),
            [input_shape[0], input_shape[1], input_shape[2]],
        ))),
        (1, 2) => {
            let reduced = reduce_to_1_from_2(
                gradient
                    .reshape([gradient.shape()[0], gradient.shape()[1]])
                    .to_concrete(),
                input_shape[0],
            );
            Ok(Box::new(reduced))
        }
        (1, 3) => {
            let reduced = reduce_to_1_from_3(
                gradient
                    .reshape([
                        gradient.shape()[0],
                        gradient.shape()[1],
                        gradient.shape()[2],
                    ])
                    .to_concrete(),
                input_shape[0],
            );
            Ok(Box::new(reduced))
        }
        (2, 3) => {
            let reduced = reduce_to_2_from_3(
                gradient
                    .reshape([
                        gradient.shape()[0],
                        gradient.shape()[1],
                        gradient.shape()[2],
                    ])
                    .to_concrete(),
                [input_shape[0], input_shape[1]],
            );
            Ok(Box::new(reduced))
        }
        _ => Err(Error::msg(
            "unsupported broadcast gradient rank combination",
        )),
    }
}

fn reduce_same_rank_broadcast_1(
    mut gradient: RawTensor<1, f32>,
    input_shape: [usize; 1],
) -> RawTensor<1, f32> {
    if input_shape[0] == 1 && gradient.shape()[0] != 1 {
        gradient = gradient.sum_keepdim::<0>(0).to_concrete();
    }
    gradient.reshape(input_shape).to_concrete()
}

fn reduce_same_rank_broadcast_2(
    mut gradient: RawTensor<2, f32>,
    input_shape: [usize; 2],
) -> RawTensor<2, f32> {
    let grad_shape = gradient.shape();
    if input_shape[0] == 1 && grad_shape[0] != 1 {
        gradient = gradient.sum_keepdim::<1>(0).to_concrete();
    }
    if input_shape[1] == 1 && grad_shape[1] != 1 {
        gradient = gradient.sum_keepdim::<1>(1).to_concrete();
    }
    gradient.reshape(input_shape).to_concrete()
}

fn reduce_same_rank_broadcast_3(
    mut gradient: RawTensor<3, f32>,
    input_shape: [usize; 3],
) -> RawTensor<3, f32> {
    let grad_shape = gradient.shape();
    if input_shape[0] == 1 && grad_shape[0] != 1 {
        gradient = gradient.sum_keepdim::<2>(0).to_concrete();
    }
    if input_shape[1] == 1 && grad_shape[1] != 1 {
        gradient = gradient.sum_keepdim::<2>(1).to_concrete();
    }
    if input_shape[2] == 1 && grad_shape[2] != 1 {
        gradient = gradient.sum_keepdim::<2>(2).to_concrete();
    }
    gradient.reshape(input_shape).to_concrete()
}

fn reduce_to_1_from_2(mut gradient: RawTensor<2, f32>, target: usize) -> RawTensor<1, f32> {
    if gradient.shape()[0] != 1 {
        gradient = gradient.sum_keepdim::<1>(0);
    }
    if gradient.shape()[1] != target {
        gradient = gradient.sum_keepdim::<1>(1);
    }
    gradient.reshape([target]).to_concrete()
}

fn reduce_to_1_from_3(mut gradient: RawTensor<3, f32>, target: usize) -> RawTensor<1, f32> {
    if gradient.shape()[0] != 1 {
        gradient = gradient.sum_keepdim::<2>(0);
    }
    if gradient.shape()[1] != 1 {
        gradient = gradient.sum_keepdim::<2>(1);
    }
    if gradient.shape()[2] != target {
        gradient = gradient.sum_keepdim::<2>(2);
    }
    gradient.reshape([target]).to_concrete()
}

fn reduce_to_2_from_3(mut gradient: RawTensor<3, f32>, target: [usize; 2]) -> RawTensor<2, f32> {
    if gradient.shape()[0] != 1 {
        gradient = gradient.sum_keepdim::<2>(0);
    }
    if gradient.shape()[1] != target[0] {
        gradient = gradient.sum_keepdim::<2>(1);
    }
    if gradient.shape()[2] != target[1] {
        gradient = gradient.sum_keepdim::<2>(2);
    }
    gradient.reshape(target).to_concrete()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToVec1, ToVec2};

    fn assert_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-3, "expected {right}, got {left}");
    }

    #[tokio::test]
    async fn test_backward_squared_sum_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let loss = x.sqr().sum();
        let gradients = loss.backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(dx[0], 2.0);
        assert_close(dx[1], 4.0);
        assert_close(dx[2], 6.0);
    }

    #[tokio::test]
    async fn test_backward_matmul_with_broadcast_bias_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let w: Tensor<2> = Tensor::new(&graph, &device, &[[0.5f32], [1.0], [1.5]]);
        let b: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

        let y = x.mat_mul(&w).add(&b.broadcast_as([2, 1]));
        let loss = y.sum(1).sum();

        let gradients = loss.backward().unwrap();
        let dw = gradients
            .get(&w)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();
        let db = gradients
            .get(&b)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec1();

        assert_close(dw[0][0], 5.0);
        assert_close(dw[1][0], 7.0);
        assert_close(dw[2][0], 9.0);
        assert_close(db[0], 2.0);
    }

    #[tokio::test]
    async fn test_backward_embedding_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let table: Tensor<2> =
            Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let indices: RawTensor<2, u32> = RawTensor::new(&device, &[[0u32, 2u32]]);
        let embedded = table.embedding(&indices);
        let loss = embedded.sum(2).sum(1).sum();

        let gradients = loss.backward().unwrap();
        let dtable = gradients
            .get(&table)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_close(dtable[0][0], 1.0);
        assert_close(dtable[0][1], 1.0);
        assert_close(dtable[1][0], 0.0);
        assert_close(dtable[1][1], 0.0);
        assert_close(dtable[2][0], 1.0);
        assert_close(dtable[2][1], 1.0);
    }

    #[tokio::test]
    async fn test_backward_gather_last_cpu() {
        let graph = Graph::new();
        let device = Device::cpu();

        let values: Tensor<2> =
            Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices: RawTensor<1, u32> = RawTensor::new(&device, &[2u32, 0u32]);
        let gathered = values.gather_last(&indices);
        let loss = gathered.sum();

        let gradients = loss.backward().unwrap();
        let dvalues = gradients
            .get(&values)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec2();

        assert_close(dvalues[0][0], 0.0);
        assert_close(dvalues[0][1], 0.0);
        assert_close(dvalues[0][2], 1.0);
        assert_close(dvalues[1][0], 1.0);
        assert_close(dvalues[1][1], 0.0);
        assert_close(dvalues[1][2], 0.0);
    }

    #[tokio::test]
    async fn test_cpu_graph_drops_after_backward() {
        let graph = Graph::new();
        let weak = Arc::downgrade(&graph.inner);
        let device = Device::cpu();

        let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let w: Tensor<2> = Tensor::new(&graph, &device, &[[0.5f32, -1.0], [1.5, 2.0]]);
        let loss = x.mat_mul(&w).sum(1).sum();
        let gradients = loss.backward().unwrap();
        assert!(gradients.get(&x).is_some());
        assert!(gradients.get(&w).is_some());

        drop(gradients);
        drop(loss);
        drop(x);
        drop(w);
        drop(graph);

        assert!(
            weak.upgrade().is_none(),
            "autograd graph stayed alive after all tensors were dropped",
        );
    }
}
