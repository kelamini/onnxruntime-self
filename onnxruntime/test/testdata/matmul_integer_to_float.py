from enum import Enum  # noqa: F401

import onnx
from onnx import TensorProto, helper


def GenerateModel(model_name, sign_i, sign_w, has_zp=True, bias=False):  # noqa: N802
    nodes = [  # subgraph
        helper.make_node(
            "MatMulInteger",
            ["A", "B", "a_zero_point", "b_zero_point"] if has_zp else ["A", "B"],
            ["matmul_output_int32"],
            "MatMulInteger",
        ),
        helper.make_node("Mul", ["a_scale", "b_scale"], ["multiplier"], "mul_right"),
        helper.make_node("Cast", ["matmul_output_int32"], ["matmul_output_float"], "cast", to=1),
        helper.make_node(
            "Mul",
            ["matmul_output_float", "multiplier"],
            ["mul_bottom_output" if bias else "Y"],
            "mul_bottom",
        ),
    ]

    inputs = [  # inputs
        helper.make_tensor_value_info("A", TensorProto.INT8 if sign_i else TensorProto.UINT8, ["M", "K"]),
        helper.make_tensor_value_info("B", TensorProto.INT8 if sign_w else TensorProto.UINT8, ["K", "N"]),
        helper.make_tensor_value_info("a_scale", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("b_scale", TensorProto.FLOAT, ["C"]),
    ]

    if has_zp:
        inputs.extend(
            [
                helper.make_tensor_value_info(
                    "a_zero_point",
                    TensorProto.INT8 if sign_i else TensorProto.UINT8,
                    [1],
                ),
                helper.make_tensor_value_info(
                    "b_zero_point",
                    TensorProto.INT8 if sign_w else TensorProto.UINT8,
                    ["C"],
                ),
            ]
        )

    if bias:
        nodes.extend([helper.make_node("Add", ["mul_bottom_output", "bias"], ["Y"], "add")])

        inputs.extend([helper.make_tensor_value_info("bias", TensorProto.FLOAT, ["N"])])

    graph = helper.make_graph(
        nodes,
        "DynamicQuantizeMatMul_fusion",  # name
        inputs,
        [  # outputs
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", "N"]),
        ],
    )

    model = helper.make_model(graph)
    onnx.save(model, model_name)


if __name__ == "__main__":
    GenerateModel("matmul_integer_to_float_int8.onnx", False, True)
    GenerateModel("matmul_integer_to_float_uint8.onnx", False, False)
    GenerateModel("matmul_integer_to_float_int8_bias.onnx", False, True, False, True)
    GenerateModel("matmul_integer_to_float_uint8_bias.onnx", False, False, False, True)

    GenerateModel("matmul_integer_to_float_int8_int8.onnx", True, True)
    GenerateModel("matmul_integer_to_float_int8_int8_bias.onnx", True, True, False, True)
