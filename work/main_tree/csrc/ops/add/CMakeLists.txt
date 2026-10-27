# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# Add算子自注册

# Kernel源文件(bisheng编译)
set(ADD_KERNEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/add_kernel.cpp
)

# Plugin源文件(g++编译)
set(ADD_PLUGIN_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/op_plugin/add_plugin.cpp
)

# 注册到全局列表
register_direct_launch_op(
    "${ADD_KERNEL_SRCS}"
    op_kernel
    "${ADD_PLUGIN_SRCS}"
    op_kernel
)