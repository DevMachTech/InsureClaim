жҐ
Є&З&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
°
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
.
Identity

input"T
output"T"	
Ttype
№
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0ю€€€€€€€€"
value_indexint(0ю€€€€€€€€"+

vocab_sizeint€€€€€€€€€(0€€€€€€€€€"
	delimiterstring	"
offsetint И
:
Less
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
TouttypeИ
2
LookupTableSizeV2
table_handle
size	И
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
Н
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint€€€€€€€€€"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Р
ParseExampleV2

serialized	
names
sparse_keys

dense_keys
ragged_keys
dense_defaults2Tdense
sparse_indices	*
num_sparse
sparse_values2sparse_types
sparse_shapes	*
num_sparse
dense_values2Tdense#
ragged_values2ragged_value_types'
ragged_row_splits2ragged_split_types"
Tdense
list(type)(:
2	"

num_sparseint("%
sparse_types
list(type)(:
2	"+
ragged_value_types
list(type)(:
2	"*
ragged_split_types
list(type)(:
2	"
dense_shapeslist(shape)(
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
-
Sqrt
x"T
y"T"
Ttype:

2
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
<
Sub
x"T
y"T
z"T"
Ttype:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
9
VarIsInitializedOp
resource
is_initialized
И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58≈…
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 
Б
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
z
Variable/AssignAssignVariableOpVariableasset_path_initializer*&
 _has_manual_control_dependencies(*
dtype0
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
Y
asset_path_initializer_1Placeholder*
_output_shapes
: *
dtype0*
shape: 
З

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
А
Variable_1/AssignAssignVariableOp
Variable_1asset_path_initializer_1*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
: *
dtype0
Y
asset_path_initializer_2Placeholder*
_output_shapes
: *
dtype0*
shape: 
З

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
А
Variable_2/AssignAssignVariableOp
Variable_2asset_path_initializer_2*&
 _has_manual_control_dependencies(*
dtype0
a
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
: *
dtype0
P
ConstConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *Ц”ЧJ
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *;+иD
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  А?
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *   А
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *  А@
L
Const_8Const*
_output_shapes
: *
dtype0*
valueB
 *  Ањ
I
Const_9Const*
_output_shapes
: *
dtype0	*
value	B	 R
S
Const_10Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
J
Const_11Const*
_output_shapes
: *
dtype0	*
value	B	 R
J
Const_12Const*
_output_shapes
: *
dtype0	*
value	B	 R
J
Const_13Const*
_output_shapes
: *
dtype0	*
value	B	 R
S
Const_14Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
J
Const_15Const*
_output_shapes
: *
dtype0	*
value	B	 R
J
Const_16Const*
_output_shapes
: *
dtype0	*
value	B	 R
J
Const_17Const*
_output_shapes
: *
dtype0	*
value	B	 R
Л
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315688
Н
StatefulPartitionedCall_1StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315688
Н
StatefulPartitionedCall_2StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315694
Н
StatefulPartitionedCall_3StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315699
Н
StatefulPartitionedCall_4StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315699
Н
StatefulPartitionedCall_5StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315705
Н
StatefulPartitionedCall_6StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315710
Н
StatefulPartitionedCall_7StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315710
Н
StatefulPartitionedCall_8StatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315716
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
~
Adam/v/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_5/bias
w
'Adam/v/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_5/bias
w
'Adam/m/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/bias*
_output_shapes
:*
dtype0
Ж
Adam/v/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/v/dense_5/kernel

)Adam/v/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_5/kernel*
_output_shapes

:*
dtype0
Ж
Adam/m/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/m/dense_5/kernel

)Adam/m/dense_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_5/kernel*
_output_shapes

:*
dtype0
~
Adam/v/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_4/bias
w
'Adam/v/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_4/bias
w
'Adam/m/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/bias*
_output_shapes
:*
dtype0
З
Adam/v/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И*&
shared_nameAdam/v/dense_4/kernel
А
)Adam/v/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_4/kernel*
_output_shapes
:	И*
dtype0
З
Adam/m/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И*&
shared_nameAdam/m/dense_4/kernel
А
)Adam/m/dense_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_4/kernel*
_output_shapes
:	И*
dtype0

Adam/v/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:И*$
shared_nameAdam/v/dense_3/bias
x
'Adam/v/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/bias*
_output_shapes	
:И*
dtype0

Adam/m/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:И*$
shared_nameAdam/m/dense_3/bias
x
'Adam/m/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/bias*
_output_shapes	
:И*
dtype0
З
Adam/v/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И*&
shared_nameAdam/v/dense_3/kernel
А
)Adam/v/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_3/kernel*
_output_shapes
:	И*
dtype0
З
Adam/m/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И*&
shared_nameAdam/m/dense_3/kernel
А
)Adam/m/dense_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_3/kernel*
_output_shapes
:	И*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

:*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	И*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:И*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:И*
dtype0
y
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	И*
shared_namedense_3/kernel
r
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes
:	И*
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
ђ
StatefulPartitionedCall_9StatefulPartitionedCallserving_default_examplesConst_8Const_7Const_6Const_5Const_4Const_3Const_2Const_1StatefulPartitionedCall_8ConstConst_17Const_16Const_15StatefulPartitionedCall_5Const_14Const_13Const_12Const_11StatefulPartitionedCall_2Const_10Const_9dense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*'
Tin 
2												*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_314341
e
ReadVariableOpReadVariableOp
Variable_2^Variable_2/Assign*
_output_shapes
: *
dtype0
џ
StatefulPartitionedCall_10StatefulPartitionedCallReadVariableOpStatefulPartitionedCall_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_315449
g
ReadVariableOp_1ReadVariableOp
Variable_2^Variable_2/Assign*
_output_shapes
: *
dtype0
Ё
StatefulPartitionedCall_11StatefulPartitionedCallReadVariableOp_1StatefulPartitionedCall_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_315488
g
ReadVariableOp_2ReadVariableOp
Variable_1^Variable_1/Assign*
_output_shapes
: *
dtype0
Ё
StatefulPartitionedCall_12StatefulPartitionedCallReadVariableOp_2StatefulPartitionedCall_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_315526
g
ReadVariableOp_3ReadVariableOp
Variable_1^Variable_1/Assign*
_output_shapes
: *
dtype0
Ё
StatefulPartitionedCall_13StatefulPartitionedCallReadVariableOp_3StatefulPartitionedCall_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_315565
c
ReadVariableOp_4ReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
Ё
StatefulPartitionedCall_14StatefulPartitionedCallReadVariableOp_4StatefulPartitionedCall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_315603
c
ReadVariableOp_5ReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
Ё
StatefulPartitionedCall_15StatefulPartitionedCallReadVariableOp_5StatefulPartitionedCall_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_315642
ф
NoOpNoOp^StatefulPartitionedCall_10^StatefulPartitionedCall_11^StatefulPartitionedCall_12^StatefulPartitionedCall_13^StatefulPartitionedCall_14^StatefulPartitionedCall_15^Variable/Assign^Variable_1/Assign^Variable_2/Assign
≈`
Const_18Const"/device:CPU:0*
_output_shapes
: *
dtype0*э_
valueу_Bр_ Bй_
г
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-0
layer-10
layer_with_weights-1
layer-11
layer_with_weights-2
layer-12
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

signatures*
* 
* 
* 
* 
* 
* 
О
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
О
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 
О
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 
О
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
¶
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
¶
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias*
¶
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias*
і
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
$N _saved_model_loader_tracked_dict* 
.
60
71
>2
?3
F4
G5*
.
60
71
>2
?3
F4
G5*
* 
∞
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_3* 
6
Xtrace_0
Ytrace_1
Ztrace_2
[trace_3* 
* 
Б
\
_variables
]_iterations
^_learning_rate
__index_dict
`
_momentums
a_velocities
b_update_step_xla*

cserving_default* 
* 
* 
* 
С
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

itrace_0* 

jtrace_0* 
* 
* 
* 
С
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses* 

ptrace_0* 

qtrace_0* 
* 
* 
* 
С
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses* 

wtrace_0* 

xtrace_0* 
* 
* 
* 
С
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

~trace_0* 

trace_0* 

60
71*

60
71*
* 
Ш
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

Еtrace_0* 

Жtrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

>0
?1*

>0
?1*
* 
Ш
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

Мtrace_0* 

Нtrace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 
Ш
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

Уtrace_0* 

Фtrace_0* 
^X
VARIABLE_VALUEdense_5/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_5/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

Ъtrace_0
Ыtrace_1* 

Ьtrace_0
Эtrace_1* 
y
Ю	_imported
Я_wrapped_function
†_structured_inputs
°_structured_outputs
Ґ_output_to_inputs_map* 
* 
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*

£0
§1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
n
]0
•1
¶2
І3
®4
©5
™6
Ђ7
ђ8
≠9
Ѓ10
ѓ11
∞12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
4
•0
І1
©2
Ђ3
≠4
ѓ5*
4
¶0
®1
™2
ђ3
Ѓ4
∞5*
V
±trace_0
≤trace_1
≥trace_2
іtrace_3
µtrace_4
ґtrace_5* 
Ђ
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ђ
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20* 
Ђ
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20* 
Ђ
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20* 
Ђ
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20* 
ђ
…created_variables
 	resources
Ћtrackable_objects
ћinitializers
Ќassets
ќ
signatures
$ѕ_self_saveable_object_factories
Яtransform_fn* 
Ђ
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20* 
* 
* 
* 
<
–	variables
—	keras_api

“total

”count*
M
‘	variables
’	keras_api

÷total

„count
Ў
_fn_kwargs*
`Z
VARIABLE_VALUEAdam/m/dense_3/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_3/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_3/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_3/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_4/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_4/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_4/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_4/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_5/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_5/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_5/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_5/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
2
ў0
Џ1
џ2
№3
Ё4
ё5* 
* 

я0
а1
б2* 

в0
г1
д2* 

еserving_default* 
* 

“0
”1*

–	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

÷0
„1*

‘	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
V
я_initializer
ж_create_resource
з_initialize
и_destroy_resource* 
V
я_initializer
й_create_resource
к_initialize
л_destroy_resource* 
V
а_initializer
м_create_resource
н_initialize
о_destroy_resource* 
V
а_initializer
п_create_resource
р_initialize
с_destroy_resource* 
V
б_initializer
т_create_resource
у_initialize
ф_destroy_resource* 
V
б_initializer
х_create_resource
ц_initialize
ч_destroy_resource* 
8
в	_filename
$ш_self_saveable_object_factories* 
8
г	_filename
$щ_self_saveable_object_factories* 
8
д	_filename
$ъ_self_saveable_object_factories* 
* 
* 
* 
Ђ
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20* 

ыtrace_0* 

ьtrace_0* 

эtrace_0* 

юtrace_0* 

€trace_0* 

Аtrace_0* 

Бtrace_0* 

Вtrace_0* 

Гtrace_0* 

Дtrace_0* 

Еtrace_0* 

Жtrace_0* 

Зtrace_0* 

Иtrace_0* 

Йtrace_0* 

Кtrace_0* 

Лtrace_0* 

Мtrace_0* 
* 
* 
* 
* 

в	capture_0* 
* 
* 

в	capture_0* 
* 
* 

г	capture_0* 
* 
* 

г	capture_0* 
* 
* 

д	capture_0* 
* 
* 

д	capture_0* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ґ	
StatefulPartitionedCall_16StatefulPartitionedCallsaver_filename"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp)Adam/m/dense_3/kernel/Read/ReadVariableOp)Adam/v/dense_3/kernel/Read/ReadVariableOp'Adam/m/dense_3/bias/Read/ReadVariableOp'Adam/v/dense_3/bias/Read/ReadVariableOp)Adam/m/dense_4/kernel/Read/ReadVariableOp)Adam/v/dense_4/kernel/Read/ReadVariableOp'Adam/m/dense_4/bias/Read/ReadVariableOp'Adam/v/dense_4/bias/Read/ReadVariableOp)Adam/m/dense_5/kernel/Read/ReadVariableOp)Adam/v/dense_5/kernel/Read/ReadVariableOp'Adam/m/dense_5/bias/Read/ReadVariableOp'Adam/v/dense_5/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst_18*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_315823
ќ
StatefulPartitionedCall_17StatefulPartitionedCallsaver_filenamedense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias	iterationlearning_rateAdam/m/dense_3/kernelAdam/v/dense_3/kernelAdam/m/dense_3/biasAdam/v/dense_3/biasAdam/m/dense_4/kernelAdam/v/dense_4/kernelAdam/m/dense_4/biasAdam/v/dense_4/biasAdam/m/dense_5/kernelAdam/v/dense_5/kernelAdam/m/dense_5/biasAdam/v/dense_5/biastotal_1count_1totalcount*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_315905∞д
тo
э
'__inference_serve_tf_examples_fn_310907
examples#
transform_features_layer_310822#
transform_features_layer_310824#
transform_features_layer_310826#
transform_features_layer_310828#
transform_features_layer_310830#
transform_features_layer_310832#
transform_features_layer_310834	#
transform_features_layer_310836	#
transform_features_layer_310838#
transform_features_layer_310840	#
transform_features_layer_310842	#
transform_features_layer_310844	#
transform_features_layer_310846	#
transform_features_layer_310848#
transform_features_layer_310850	#
transform_features_layer_310852	#
transform_features_layer_310854	#
transform_features_layer_310856	#
transform_features_layer_310858#
transform_features_layer_310860	#
transform_features_layer_310862	A
.model_1_dense_3_matmul_readvariableop_resource:	И>
/model_1_dense_3_biasadd_readvariableop_resource:	ИA
.model_1_dense_4_matmul_readvariableop_resource:	И=
/model_1_dense_4_biasadd_readvariableop_resource:@
.model_1_dense_5_matmul_readvariableop_resource:=
/model_1_dense_5_biasadd_readvariableop_resource:
identityИҐ&model_1/dense_3/BiasAdd/ReadVariableOpҐ%model_1/dense_3/MatMul/ReadVariableOpҐ&model_1/dense_4/BiasAdd/ReadVariableOpҐ%model_1/dense_4/MatMul/ReadVariableOpҐ&model_1/dense_5/BiasAdd/ReadVariableOpҐ%model_1/dense_5/MatMul/ReadVariableOpҐ0transform_features_layer/StatefulPartitionedCallU
ParseExample/ConstConst*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_1Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_2Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_3Const*
_output_shapes
: *
dtype0	*
valueB	 W
ParseExample/Const_4Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_5Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_6Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_7Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_8Const*
_output_shapes
: *
dtype0*
valueB W
ParseExample/Const_9Const*
_output_shapes
: *
dtype0*
valueB d
!ParseExample/ParseExampleV2/namesConst*
_output_shapes
: *
dtype0*
valueB j
'ParseExample/ParseExampleV2/sparse_keysConst*
_output_shapes
: *
dtype0*
valueB В
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:
*
dtype0*І
valueЭBЪ
BBuilding DimensionBBuilding_FencedBBuilding_PaintedBBuilding_TypeBCustomer IdBGardenBGeo_CodeBInsured_PeriodBNumberOfWindowsB
Settlementj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB «
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0ParseExample/Const_4:output:0ParseExample/Const_5:output:0ParseExample/Const_6:output:0ParseExample/Const_7:output:0ParseExample/Const_8:output:0ParseExample/Const_9:output:0*
Tdense
2
	*‘
_output_shapesЅ
Њ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€*N
dense_shapes>
<::::::::::*

num_sparse *
ragged_split_types
 *
ragged_value_types
 *
sparse_types
 x
transform_features_layer/ShapeShape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:v
,transform_features_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
.transform_features_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:x
.transform_features_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ќ
&transform_features_layer/strided_sliceStridedSlice'transform_features_layer/Shape:output:05transform_features_layer/strided_slice/stack:output:07transform_features_layer/strided_slice/stack_1:output:07transform_features_layer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
 transform_features_layer/Shape_1Shape*ParseExample/ParseExampleV2:dense_values:0*
T0*
_output_shapes
:x
.transform_features_layer/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0transform_features_layer/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0transform_features_layer/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(transform_features_layer/strided_slice_1StridedSlice)transform_features_layer/Shape_1:output:07transform_features_layer/strided_slice_1/stack:output:09transform_features_layer/strided_slice_1/stack_1:output:09transform_features_layer/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'transform_features_layer/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ј
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R Ј
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:€€€€€€€€€∆
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€Љ
0transform_features_layer/StatefulPartitionedCallStatefulPartitionedCall*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:1*ParseExample/ParseExampleV2:dense_values:2*ParseExample/ParseExampleV2:dense_values:38transform_features_layer/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:4*ParseExample/ParseExampleV2:dense_values:5*ParseExample/ParseExampleV2:dense_values:6*ParseExample/ParseExampleV2:dense_values:7*ParseExample/ParseExampleV2:dense_values:8*ParseExample/ParseExampleV2:dense_values:9transform_features_layer_310822transform_features_layer_310824transform_features_layer_310826transform_features_layer_310828transform_features_layer_310830transform_features_layer_310832transform_features_layer_310834transform_features_layer_310836transform_features_layer_310838transform_features_layer_310840transform_features_layer_310842transform_features_layer_310844transform_features_layer_310846transform_features_layer_310848transform_features_layer_310850transform_features_layer_310852transform_features_layer_310854transform_features_layer_310856transform_features_layer_310858transform_features_layer_310860transform_features_layer_310862*+
Tin$
"2 														*
Tout
	2	*Н
_output_shapes{
y:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *"
fR
__inference_pruned_310515a
model_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ѓ
model_1/ExpandDims
ExpandDims9transform_features_layer/StatefulPartitionedCall:output:3model_1/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
model_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
model_1/ExpandDims_1
ExpandDims9transform_features_layer/StatefulPartitionedCall:output:5!model_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
model_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≤
model_1/ExpandDims_2
ExpandDims9transform_features_layer/StatefulPartitionedCall:output:0!model_1/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
!model_1/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ћ
model_1/concatenate_4/concatConcatV2model_1/ExpandDims:output:0model_1/ExpandDims_1:output:0*model_1/concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€c
!model_1/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ј
model_1/concatenate_5/concatConcatV29transform_features_layer/StatefulPartitionedCall:output:19transform_features_layer/StatefulPartitionedCall:output:29transform_features_layer/StatefulPartitionedCall:output:6*model_1/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€i
'model_1/concatenate_6/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :А
#model_1/concatenate_6/concat/concatIdentitymodel_1/ExpandDims_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€c
!model_1/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Л
model_1/concatenate_7/concatConcatV2%model_1/concatenate_4/concat:output:0%model_1/concatenate_5/concat:output:0,model_1/concatenate_6/concat/concat:output:0*model_1/concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Х
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype0©
model_1/dense_3/MatMulMatMul%model_1/concatenate_7/concat:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ИУ
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype0І
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Иq
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ИХ
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype0•
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€p
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0•
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€v
model_1/dense_5/SigmoidSigmoid model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
IdentityIdentitymodel_1/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€м
NoOpNoOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp1^transform_features_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp2d
0transform_features_layer/StatefulPartitionedCall0transform_features_layer/StatefulPartitionedCall:M I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ю

х
C__inference_dense_4_layer_call_and_return_conditional_losses_314760

inputs1
matmul_readvariableop_resource:	И-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€И: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€И
 
_user_specified_nameinputs
Л
V
)__inference_restored_function_body_315582
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310590^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
ї
9
)__inference_restored_function_body_315649
identity–
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__destroyer_310325O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
™
H
__inference__creator_315585
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315582^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
ї
9
)__inference_restored_function_body_315456
identity–
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__destroyer_310575O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
¶
;
__inference__creator_310355
identityИҐ
hash_table¬

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*Ќ
shared_nameљЇhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/43/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_vocabulary', shape=(), dtype=string)_-2_-1_load_310305_310351*
use_node_name_sharing(*
value_dtype0	S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
≈Ј
 
__inference_pruned_310515

inputs
inputs_1
inputs_2
inputs_3	
inputs_4	
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10-
)scale_to_0_1_min_and_max_identity_2_input-
)scale_to_0_1_min_and_max_identity_3_input/
+scale_to_0_1_1_min_and_max_identity_2_input/
+scale_to_0_1_1_min_and_max_identity_3_input0
,scale_to_z_score_mean_and_var_identity_input2
.scale_to_z_score_mean_and_var_identity_1_input:
6compute_and_apply_vocabulary_vocabulary_identity_input	<
8compute_and_apply_vocabulary_vocabulary_identity_1_input	c
_compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handled
`compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value	2
.compute_and_apply_vocabulary_apply_vocab_sub_x	<
8compute_and_apply_vocabulary_1_vocabulary_identity_input	>
:compute_and_apply_vocabulary_1_vocabulary_identity_1_input	e
acompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handlef
bcompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value	4
0compute_and_apply_vocabulary_1_apply_vocab_sub_x	<
8compute_and_apply_vocabulary_2_vocabulary_identity_input	>
:compute_and_apply_vocabulary_2_vocabulary_identity_1_input	e
acompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handlef
bcompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value	4
0compute_and_apply_vocabulary_2_apply_vocab_sub_x	
identity

identity_1

identity_2

identity_3

identity_4	

identity_5

identity_6И`
scale_to_z_score/NotEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€c
 scale_to_0_1_1/min_and_max/ShapeConst*
_output_shapes
: *
dtype0*
valueB e
"scale_to_0_1_1/min_and_max/Shape_1Const*
_output_shapes
: *
dtype0*
valueB y
/scale_to_0_1_1/min_and_max/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ™
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:®
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)x (scale_to_0_1_1/min_and_max/Shape:0) = ™
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*<
value3B1 B+y (scale_to_0_1_1/min_and_max/Shape_1:0) = a
scale_to_0_1/min_and_max/ShapeConst*
_output_shapes
: *
dtype0*
valueB c
 scale_to_0_1/min_and_max/Shape_1Const*
_output_shapes
: *
dtype0*
valueB w
-scale_to_0_1/min_and_max/assert_equal_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: ®
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:§
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*8
value/B- B'x (scale_to_0_1/min_and_max/Shape:0) = ¶
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*:
value1B/ B)y (scale_to_0_1/min_and_max/Shape_1:0) = O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  А?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    `
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Q
one_hot_1/depthConst*
_output_shapes
: *
dtype0*
value	B :W
one_hot_1/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  А?X
one_hot_1/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    `
Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   e
 scale_to_0_1/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    W
scale_to_0_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?Y
scale_to_0_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    `
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€g
"scale_to_0_1_1/min_and_max/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
scale_to_0_1_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?[
scale_to_0_1_1/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    b
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
€€€€€€€€€Q
one_hot_2/depthConst*
_output_shapes
: *
dtype0*
value	B :W
one_hot_2/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  А?X
one_hot_2/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    `
Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"€€€€   Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€Б
&scale_to_z_score/mean_and_var/IdentityIdentity,scale_to_z_score_mean_and_var_identity_input*
T0*
_output_shapes
: Ф
scale_to_z_score/subSubinputs_copy:output:0/scale_to_z_score/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:€€€€€€€€€t
scale_to_z_score/zeros_like	ZerosLikescale_to_z_score/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€Е
(scale_to_z_score/mean_and_var/Identity_1Identity.scale_to_z_score_mean_and_var_identity_1_input*
T0*
_output_shapes
: q
scale_to_z_score/SqrtSqrt1scale_to_z_score/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: З
scale_to_z_score/NotEqualNotEqualscale_to_z_score/Sqrt:y:0$scale_to_z_score/NotEqual/y:output:0*
T0*
_output_shapes
: l
scale_to_z_score/CastCastscale_to_z_score/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: Л
scale_to_z_score/addAddV2scale_to_z_score/zeros_like:y:0scale_to_z_score/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€z
scale_to_z_score/Cast_1Castscale_to_z_score/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€К
scale_to_z_score/truedivRealDivscale_to_z_score/sub:z:0scale_to_z_score/Sqrt:y:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
scale_to_z_score/SelectV2SelectV2scale_to_z_score/Cast_1:y:0scale_to_z_score/truediv:z:0scale_to_z_score/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€А
	Reshape_2Reshape"scale_to_z_score/SelectV2:output:0Reshape_2/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:€€€€€€€€€Ы
Rcompute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2_compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleinputs_1_copy:output:0`compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*&
 _has_manual_control_dependencies(*
_output_shapes
:џ
Pcompute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2_compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleS^compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2*&
 _has_manual_control_dependencies(*
_output_shapes
: ≥
/scale_to_0_1_1/min_and_max/assert_equal_1/EqualEqual)scale_to_0_1_1/min_and_max/Shape:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: ї
-scale_to_0_1_1/min_and_max/assert_equal_1/AllAll3scale_to_0_1_1/min_and_max/assert_equal_1/Equal:z:08scale_to_0_1_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: ≠
-scale_to_0_1/min_and_max/assert_equal_1/EqualEqual'scale_to_0_1/min_and_max/Shape:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: µ
+scale_to_0_1/min_and_max/assert_equal_1/AllAll1scale_to_0_1/min_and_max/assert_equal_1/Equal:z:06scale_to_0_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: м
5scale_to_0_1/min_and_max/assert_equal_1/Assert/AssertAssert4scale_to_0_1/min_and_max/assert_equal_1/All:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0'scale_to_0_1/min_and_max/Shape:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T	
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ≤
7scale_to_0_1_1/min_and_max/assert_equal_1/Assert/AssertAssert6scale_to_0_1_1/min_and_max/assert_equal_1/All:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0)scale_to_0_1_1/min_and_max/Shape:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:06^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert*
T	
2*&
 _has_manual_control_dependencies(*
_output_shapes
 U
inputs_2_copyIdentityinputs_2*
T0*'
_output_shapes
:€€€€€€€€€°
Tcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2acompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleinputs_2_copy:output:0bcompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*&
 _has_manual_control_dependencies(*
_output_shapes
:б
Rcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2acompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleU^compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2*&
 _has_manual_control_dependencies(*
_output_shapes
: W
inputs_10_copyIdentity	inputs_10*
T0*'
_output_shapes
:€€€€€€€€€Ґ
Tcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2acompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleinputs_10_copy:output:0bcompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*&
 _has_manual_control_dependencies(*
_output_shapes
:б
Rcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2acompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleU^compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2*&
 _has_manual_control_dependencies(*
_output_shapes
: а
NoOpNoOpS^compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2Q^compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2U^compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2S^compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2U^compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2S^compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Size/LookupTableSizeV26^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert8^scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert*"
_acd_function_control_output(*&
 _has_manual_control_dependencies(*
_output_shapes
 ]
IdentityIdentityReshape_2:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€Ђ
=compute_and_apply_vocabulary/apply_vocab/None_Lookup/NotEqualNotEqual[compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0`compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*
_output_shapes
:Ѓ
@compute_and_apply_vocabulary/apply_vocab/None_Lookup/hash_bucketStringToHashBucketFastinputs_1_copy:output:0*'
_output_shapes
:€€€€€€€€€*
num_bucketsЧ
8compute_and_apply_vocabulary/apply_vocab/None_Lookup/AddAddV2Icompute_and_apply_vocabulary/apply_vocab/None_Lookup/hash_bucket:output:0Wcompute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*'
_output_shapes
:€€€€€€€€€ 
=compute_and_apply_vocabulary/apply_vocab/None_Lookup/SelectV2SelectV2Acompute_and_apply_vocabulary/apply_vocab/None_Lookup/NotEqual:z:0[compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0<compute_and_apply_vocabulary/apply_vocab/None_Lookup/Add:z:0*
T0	*
_output_shapes
:Ћ
one_hotOneHotFcompute_and_apply_vocabulary/apply_vocab/None_Lookup/SelectV2:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
_output_shapes
:r
	Reshape_3Reshapeone_hot:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_1IdentityReshape_3:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€±
?compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/NotEqualNotEqual]compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0bcompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*
_output_shapes
:∞
Bcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/hash_bucketStringToHashBucketFastinputs_2_copy:output:0*'
_output_shapes
:€€€€€€€€€*
num_bucketsЭ
:compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/AddAddV2Kcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/hash_bucket:output:0Ycompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*'
_output_shapes
:€€€€€€€€€“
?compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/SelectV2SelectV2Ccompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/NotEqual:z:0]compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0>compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/Add:z:0*
T0	*
_output_shapes
:’
	one_hot_1OneHotHcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/SelectV2:output:0one_hot_1/depth:output:0one_hot_1/on_value:output:0one_hot_1/off_value:output:0*
T0*
_output_shapes
:t
	Reshape_4Reshapeone_hot_1:output:0Reshape_4/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_2IdentityReshape_4:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€U
inputs_3_copyIdentityinputs_3*
T0	*'
_output_shapes
:€€€€€€€€€r
scale_to_0_1/CastCastinputs_3_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€{
#scale_to_0_1/min_and_max/Identity_2Identity)scale_to_0_1_min_and_max_identity_2_input*
T0*
_output_shapes
: Я
scale_to_0_1/min_and_max/sub_1Sub)scale_to_0_1/min_and_max/sub_1/x:output:0,scale_to_0_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: Д
scale_to_0_1/subSubscale_to_0_1/Cast:y:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€l
scale_to_0_1/zeros_like	ZerosLikescale_to_0_1/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€{
#scale_to_0_1/min_and_max/Identity_3Identity)scale_to_0_1_min_and_max_identity_3_input*
T0*
_output_shapes
: М
scale_to_0_1/LessLess"scale_to_0_1/min_and_max/sub_1:z:0,scale_to_0_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: b
scale_to_0_1/Cast_1Castscale_to_0_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: Б
scale_to_0_1/addAddV2scale_to_0_1/zeros_like:y:0scale_to_0_1/Cast_1:y:0*
T0*'
_output_shapes
:€€€€€€€€€r
scale_to_0_1/Cast_2Castscale_to_0_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€М
scale_to_0_1/sub_1Sub,scale_to_0_1/min_and_max/Identity_3:output:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 
scale_to_0_1/truedivRealDivscale_to_0_1/sub:z:0scale_to_0_1/sub_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
scale_to_0_1/SigmoidSigmoidscale_to_0_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€†
scale_to_0_1/SelectV2SelectV2scale_to_0_1/Cast_2:y:0scale_to_0_1/truediv:z:0scale_to_0_1/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€Ж
scale_to_0_1/mulMulscale_to_0_1/SelectV2:output:0scale_to_0_1/mul/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€В
scale_to_0_1/add_1AddV2scale_to_0_1/mul:z:0scale_to_0_1/add_1/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
ReshapeReshapescale_to_0_1/add_1:z:0Reshape/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€]

Identity_3IdentityReshape:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€U
inputs_4_copyIdentityinputs_4*
T0	*'
_output_shapes
:€€€€€€€€€g

Identity_4Identityinputs_4_copy:output:0^NoOp*
T0	*'
_output_shapes
:€€€€€€€€€U
inputs_8_copyIdentityinputs_8*
T0*'
_output_shapes
:€€€€€€€€€
%scale_to_0_1_1/min_and_max/Identity_2Identity+scale_to_0_1_1_min_and_max_identity_2_input*
T0*
_output_shapes
: •
 scale_to_0_1_1/min_and_max/sub_1Sub+scale_to_0_1_1/min_and_max/sub_1/x:output:0.scale_to_0_1_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: Й
scale_to_0_1_1/subSubinputs_8_copy:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€p
scale_to_0_1_1/zeros_like	ZerosLikescale_to_0_1_1/sub:z:0*
T0*'
_output_shapes
:€€€€€€€€€
%scale_to_0_1_1/min_and_max/Identity_3Identity+scale_to_0_1_1_min_and_max_identity_3_input*
T0*
_output_shapes
: Т
scale_to_0_1_1/LessLess$scale_to_0_1_1/min_and_max/sub_1:z:0.scale_to_0_1_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: d
scale_to_0_1_1/CastCastscale_to_0_1_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: Е
scale_to_0_1_1/addAddV2scale_to_0_1_1/zeros_like:y:0scale_to_0_1_1/Cast:y:0*
T0*'
_output_shapes
:€€€€€€€€€v
scale_to_0_1_1/Cast_1Castscale_to_0_1_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:€€€€€€€€€Т
scale_to_0_1_1/sub_1Sub.scale_to_0_1_1/min_and_max/Identity_3:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: Е
scale_to_0_1_1/truedivRealDivscale_to_0_1_1/sub:z:0scale_to_0_1_1/sub_1:z:0*
T0*'
_output_shapes
:€€€€€€€€€k
scale_to_0_1_1/SigmoidSigmoidinputs_8_copy:output:0*
T0*'
_output_shapes
:€€€€€€€€€®
scale_to_0_1_1/SelectV2SelectV2scale_to_0_1_1/Cast_1:y:0scale_to_0_1_1/truediv:z:0scale_to_0_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:€€€€€€€€€М
scale_to_0_1_1/mulMul scale_to_0_1_1/SelectV2:output:0scale_to_0_1_1/mul/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
scale_to_0_1_1/add_1AddV2scale_to_0_1_1/mul:z:0scale_to_0_1_1/add_1/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
	Reshape_1Reshapescale_to_0_1_1/add_1:z:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€_

Identity_5IdentityReshape_1:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€±
?compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/NotEqualNotEqual]compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0bcompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*
_output_shapes
:±
Bcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/hash_bucketStringToHashBucketFastinputs_10_copy:output:0*'
_output_shapes
:€€€€€€€€€*
num_bucketsЭ
:compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/AddAddV2Kcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/hash_bucket:output:0Ycompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*'
_output_shapes
:€€€€€€€€€“
?compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/SelectV2SelectV2Ccompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/NotEqual:z:0]compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0>compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/Add:z:0*
T0	*
_output_shapes
:’
	one_hot_2OneHotHcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/SelectV2:output:0one_hot_2/depth:output:0one_hot_2/on_value:output:0one_hot_2/off_value:output:0*
T0*
_output_shapes
:t
	Reshape_5Reshapeone_hot_2:output:0Reshape_5/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€c

Identity_6IdentityReshape_5:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapesю
ы:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : :- )
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-)
'
_output_shapes
:€€€€€€€€€:-	)
'
_output_shapes
:€€€€€€€€€:-
)
'
_output_shapes
:€€€€€€€€€:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ѓ
Z
.__inference_concatenate_4_layer_call_fn_315166
inputs_0
inputs_1
identityƒ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_314702`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1
®
;
__inference__creator_310590
identityИҐ
hash_tableƒ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*ѕ
shared_nameњЉhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/43/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_2_vocabulary', shape=(), dtype=string)_-2_-1_load_310305_310586*
use_node_name_sharing(*
value_dtype0	S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
љ
h
__inference__initializer_315642
unknown
	unknown_0
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315634G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
ґ5
у	
__inference__traced_save_315823
file_prefix-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop4
0savev2_adam_m_dense_3_kernel_read_readvariableop4
0savev2_adam_v_dense_3_kernel_read_readvariableop2
.savev2_adam_m_dense_3_bias_read_readvariableop2
.savev2_adam_v_dense_3_bias_read_readvariableop4
0savev2_adam_m_dense_4_kernel_read_readvariableop4
0savev2_adam_v_dense_4_kernel_read_readvariableop2
.savev2_adam_m_dense_4_bias_read_readvariableop2
.savev2_adam_v_dense_4_bias_read_readvariableop4
0savev2_adam_m_dense_5_kernel_read_readvariableop4
0savev2_adam_v_dense_5_kernel_read_readvariableop2
.savev2_adam_m_dense_5_bias_read_readvariableop2
.savev2_adam_v_dense_5_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const_18

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ъ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£

valueЩ
BЦ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЯ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B Ч

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop0savev2_adam_m_dense_3_kernel_read_readvariableop0savev2_adam_v_dense_3_kernel_read_readvariableop.savev2_adam_m_dense_3_bias_read_readvariableop.savev2_adam_v_dense_3_bias_read_readvariableop0savev2_adam_m_dense_4_kernel_read_readvariableop0savev2_adam_v_dense_4_kernel_read_readvariableop.savev2_adam_m_dense_4_bias_read_readvariableop.savev2_adam_v_dense_4_bias_read_readvariableop0savev2_adam_m_dense_5_kernel_read_readvariableop0savev2_adam_v_dense_5_kernel_read_readvariableop.savev2_adam_m_dense_5_bias_read_readvariableop.savev2_adam_v_dense_5_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const_18"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Њ
_input_shapesђ
©: :	И:И:	И:::: : :	И:	И:И:И:	И:	И::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	И:!

_output_shapes	
:И:%!

_output_shapes
:	И: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :%	!

_output_shapes
:	И:%
!

_output_shapes
:	И:!

_output_shapes	
:И:!

_output_shapes	
:И:%!

_output_shapes
:	И:%!

_output_shapes
:	И: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
®
√
__inference__initializer_310317!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityИҐ,text_file_init/InitializeTableFromTextFileV2у
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexю€€€€€€€€*
value_index€€€€€€€€€G
ConstConst*
_output_shapes
: *
dtype0*
value	B :u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
з
r
)__inference_restored_function_body_315480
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_310311^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
Л
V
)__inference_restored_function_body_315467
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310360^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
®
√
__inference__initializer_310581!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityИҐ,text_file_init/InitializeTableFromTextFileV2у
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexю€€€€€€€€*
value_index€€€€€€€€€G
ConstConst*
_output_shapes
: *
dtype0*
value	B :u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
ї
P
#__inference__update_step_xla_315145
gradient
variable:	И*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	И: *
	_noinline(:I E

_output_shapes
:	И
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ї
s
I__inference_concatenate_4_layer_call_and_return_conditional_losses_314702

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
®
;
__inference__creator_310601
identityИҐ
hash_tableƒ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*ѕ
shared_nameњЉhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/43/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_2_vocabulary', shape=(), dtype=string)_-2_-1_load_310305_310597*
use_node_name_sharing(*
value_dtype0	S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
Ы
-
__inference__destroyer_310585
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
®
;
__inference__creator_310611
identityИҐ
hash_tableƒ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*ѕ
shared_nameњЉhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/43/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_1_vocabulary', shape=(), dtype=string)_-2_-1_load_310305_310607*
use_node_name_sharing(*
value_dtype0	S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
с
ћ
(__inference_model_1_layer_call_fn_315032
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown:	И
	unknown_0:	И
	unknown_1:	И
	unknown_2:
	unknown_3:
	unknown_4:
identityИҐStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_314784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5
ї
9
)__inference_restored_function_body_315533
identity–
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__destroyer_310567O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Ґ

ц
C__inference_dense_3_layer_call_and_return_conditional_losses_314743

inputs1
matmul_readvariableop_resource:	И.
biasadd_readvariableop_resource:	И
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Иs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ИQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Иb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Иw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Щ
-
__inference__destroyer_315653
identityы
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315649G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
б
V
)__inference_restored_function_body_315699
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310611^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
Ѓ
h
.__inference_concatenate_5_layer_call_fn_315180
inputs_0
inputs_1
inputs_2
identityѕ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_314712`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2
Ы
-
__inference__destroyer_310575
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Ѓ
h
.__inference_concatenate_7_layer_call_fn_315206
inputs_0
inputs_1
inputs_2
identityѕ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_314730`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2
љ.
ѓ
!__inference__wrapped_model_314380
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholderA
.model_1_dense_3_matmul_readvariableop_resource:	И>
/model_1_dense_3_biasadd_readvariableop_resource:	ИA
.model_1_dense_4_matmul_readvariableop_resource:	И=
/model_1_dense_4_biasadd_readvariableop_resource:@
.model_1_dense_5_matmul_readvariableop_resource:=
/model_1_dense_5_biasadd_readvariableop_resource:
identityИҐ&model_1/dense_3/BiasAdd/ReadVariableOpҐ%model_1/dense_3/MatMul/ReadVariableOpҐ&model_1/dense_4/BiasAdd/ReadVariableOpҐ%model_1/dense_4/MatMul/ReadVariableOpҐ&model_1/dense_5/BiasAdd/ReadVariableOpҐ%model_1/dense_5/MatMul/ReadVariableOpc
!model_1/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :і
model_1/concatenate_4/concatConcatV2building_type_xfinsured_period_xf*model_1/concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€c
!model_1/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :«
model_1/concatenate_5/concatConcatV2building_fenced_xfbuilding_painted_xfsettlement_xf*model_1/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€i
'model_1/concatenate_6/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :n
#model_1/concatenate_6/concat/concatIdentityplaceholder*
T0*'
_output_shapes
:€€€€€€€€€c
!model_1/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Л
model_1/concatenate_7/concatConcatV2%model_1/concatenate_4/concat:output:0%model_1/concatenate_5/concat:output:0,model_1/concatenate_6/concat/concat:output:0*model_1/concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Х
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype0©
model_1/dense_3/MatMulMatMul%model_1/concatenate_7/concat:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ИУ
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype0І
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Иq
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ИХ
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype0•
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€p
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0•
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Т
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¶
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€v
model_1/dense_5/SigmoidSigmoid model_1/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
IdentityIdentitymodel_1/dense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€є
NoOpNoOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameBuilding Dimension_xf
Щ
-
__inference__destroyer_315460
identityы
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315456G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
¬
Г
I__inference_concatenate_5_layer_call_and_return_conditional_losses_315188
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2
¬
u
I__inference_concatenate_4_layer_call_and_return_conditional_losses_315173
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1
Ґ#
Б
C__inference_model_1_layer_call_and_return_conditional_losses_315008
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder!
dense_3_314992:	И
dense_3_314994:	И!
dense_4_314997:	И
dense_4_314999: 
dense_5_315002:
dense_5_315004:
identityИҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallг
concatenate_4/PartitionedCallPartitionedCallbuilding_type_xfinsured_period_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_314702ч
concatenate_5/PartitionedCallPartitionedCallbuilding_fenced_xfbuilding_painted_xfsettlement_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_314712 
concatenate_6/PartitionedCallPartitionedCallplaceholder*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_314720Ј
concatenate_7/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0&concatenate_5/PartitionedCall:output:0&concatenate_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_314730Р
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_3_314992dense_3_314994*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_314743С
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_314997dense_4_314999*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_314760С
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_315002dense_5_315004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_314777w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ђ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameBuilding Dimension_xf
≥
e
I__inference_concatenate_6_layer_call_and_return_conditional_losses_314720

inputs
identityS
concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :S
concat/concatIdentityinputs*
T0*'
_output_shapes
:€€€€€€€€€^
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
љ
h
__inference__initializer_315449
unknown
	unknown_0
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315441G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
–f
Ґ
"__inference__traced_restore_315905
file_prefix2
assignvariableop_dense_3_kernel:	И.
assignvariableop_1_dense_3_bias:	И4
!assignvariableop_2_dense_4_kernel:	И-
assignvariableop_3_dense_4_bias:3
!assignvariableop_4_dense_5_kernel:-
assignvariableop_5_dense_5_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: ;
(assignvariableop_8_adam_m_dense_3_kernel:	И;
(assignvariableop_9_adam_v_dense_3_kernel:	И6
'assignvariableop_10_adam_m_dense_3_bias:	И6
'assignvariableop_11_adam_v_dense_3_bias:	И<
)assignvariableop_12_adam_m_dense_4_kernel:	И<
)assignvariableop_13_adam_v_dense_4_kernel:	И5
'assignvariableop_14_adam_m_dense_4_bias:5
'assignvariableop_15_adam_v_dense_4_bias:;
)assignvariableop_16_adam_m_dense_5_kernel:;
)assignvariableop_17_adam_v_dense_5_kernel:5
'assignvariableop_18_adam_m_dense_5_bias:5
'assignvariableop_19_adam_v_dense_5_bias:%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_3ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9э

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*£

valueЩ
BЦ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHҐ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B Ы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOpAssignVariableOpassignvariableop_dense_3_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_3_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_4_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Є
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_5_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ґ
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_5_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:≥
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_8AssignVariableOp(assignvariableop_8_adam_m_dense_3_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_v_dense_3_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_m_dense_3_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_v_dense_3_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_m_dense_4_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_v_dense_4_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_m_dense_4_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_v_dense_4_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_m_dense_5_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_v_dense_5_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_m_dense_5_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_v_dense_5_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 я
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: ћ
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ј0
Ф
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_315422
inputs_building_dimension
inputs_building_fenced
inputs_building_painted
inputs_building_type	
inputs_customer_id
inputs_garden
inputs_geo_code
inputs_insured_period
inputs_numberofwindows
inputs_settlement
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5	
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9	

unknown_10	

unknown_11	

unknown_12

unknown_13	

unknown_14	

unknown_15	

unknown_16	

unknown_17

unknown_18	

unknown_19	
identity

identity_1

identity_2

identity_3

identity_4

identity_5ИҐStatefulPartitionedCallN
ShapeShapeinputs_building_dimension*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
Shape_1Shapeinputs_building_dimension*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€д
StatefulPartitionedCallStatefulPartitionedCallinputs_building_dimensioninputs_building_fencedinputs_building_paintedinputs_building_typePlaceholderWithDefault:output:0inputs_customer_idinputs_gardeninputs_geo_codeinputs_insured_periodinputs_numberofwindowsinputs_settlementunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*+
Tin$
"2 														*
Tout
	2	*Н
_output_shapes{
y:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *"
fR
__inference_pruned_310515k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€m

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:€€€€€€€€€m

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*э
_input_shapesл
и:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:€€€€€€€€€
3
_user_specified_nameinputs_building_dimension:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_nameinputs_building_fenced:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs_building_painted:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nameinputs_building_type:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameinputs_customer_id:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinputs_garden:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_geo_code:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameinputs_insured_period:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_nameinputs_numberofwindows:Z	V
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs_settlement:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ї
9
)__inference_restored_function_body_315495
identity–
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__destroyer_310571O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
™
H
__inference__creator_315547
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315544^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
Е
ъ
(__inference_model_1_layer_call_fn_314952
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder
unknown:	И
	unknown_0:	И
	unknown_1:	И
	unknown_2:
	unknown_3:
	unknown_4:
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallbuilding_type_xfinsured_period_xfbuilding_fenced_xfbuilding_painted_xfsettlement_xfplaceholderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_314915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameBuilding Dimension_xf
ђ
K
#__inference__update_step_xla_315150
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ю

х
C__inference_dense_4_layer_call_and_return_conditional_losses_315254

inputs1
matmul_readvariableop_resource:	И-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€И: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€И
 
_user_specified_nameinputs
Щ

ф
C__inference_dense_5_layer_call_and_return_conditional_losses_314777

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ю!
—
C__inference_model_1_layer_call_and_return_conditional_losses_314784

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5!
dense_3_314744:	И
dense_3_314746:	И!
dense_4_314761:	И
dense_4_314763: 
dense_5_314778:
dense_5_314780:
identityИҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCall–
concatenate_4/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_314702Ё
concatenate_5/PartitionedCallPartitionedCallinputs_2inputs_3inputs_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_314712«
concatenate_6/PartitionedCallPartitionedCallinputs_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_314720Ј
concatenate_7/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0&concatenate_5/PartitionedCall:output:0&concatenate_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_314730Р
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_3_314744dense_3_314746*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_314743С
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_314761dense_4_314763*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_314760С
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_314778dense_5_314780*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_314777w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ђ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
¶
;
__inference__creator_310360
identityИҐ
hash_table¬

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*Ќ
shared_nameљЇhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/43/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_vocabulary', shape=(), dtype=string)_-2_-1_load_310305_310356*
use_node_name_sharing(*
value_dtype0	S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
ї
9
)__inference_restored_function_body_315610
identity–
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__destroyer_310585O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Щ

ф
C__inference_dense_5_layer_call_and_return_conditional_losses_315274

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
™
H
__inference__creator_315431
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315428^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
б
V
)__inference_restored_function_body_315710
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310360^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
б
V
)__inference_restored_function_body_315716
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310355^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
Є
O
#__inference__update_step_xla_315155
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:H D

_output_shapes

:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
њ.
…
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_314678
placeholder
building_fenced
building_painted
building_type	
placeholder_1

garden
geo_code
insured_period
numberofwindows

settlement
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5	
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9	

unknown_10	

unknown_11	

unknown_12

unknown_13	

unknown_14	

unknown_15	

unknown_16	

unknown_17

unknown_18	

unknown_19	
identity

identity_1

identity_2

identity_3

identity_4

identity_5ИҐStatefulPartitionedCall@
ShapeShapeplaceholder*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskB
Shape_1Shapeplaceholder*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€Щ
StatefulPartitionedCallStatefulPartitionedCallplaceholderbuilding_fencedbuilding_paintedbuilding_typePlaceholderWithDefault:output:0placeholder_1gardengeo_codeinsured_periodnumberofwindows
settlementunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*+
Tin$
"2 														*
Tout
	2	*Н
_output_shapes{
y:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *"
fR
__inference_pruned_310515k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€m

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:€€€€€€€€€m

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*э
_input_shapesл
и:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameBuilding Dimension:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameBuilding_Fenced:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameBuilding_Painted:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameBuilding_Type:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_nameCustomer Id:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameGarden:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
Geo_Code:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameInsured_Period:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameNumberOfWindows:S	O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
Settlement:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ј
Б
I__inference_concatenate_5_layer_call_and_return_conditional_losses_314712

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Е
ъ
(__inference_model_1_layer_call_fn_314799
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder
unknown:	И
	unknown_0:	И
	unknown_1:	И
	unknown_2:
	unknown_3:
	unknown_4:
identityИҐStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallbuilding_type_xfinsured_period_xfbuilding_fenced_xfbuilding_painted_xfsettlement_xfplaceholderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_314784o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameBuilding Dimension_xf
љ
h
__inference__initializer_315526
unknown
	unknown_0
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315518G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
Л
V
)__inference_restored_function_body_315505
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310606^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
Ґ

ц
C__inference_dense_3_layer_call_and_return_conditional_losses_315234

inputs1
matmul_readvariableop_resource:	И.
biasadd_readvariableop_resource:	И
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	И*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Иs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:И*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ИQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Иb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Иw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
б
V
)__inference_restored_function_body_315694
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310590^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
љ
h
__inference__initializer_315565
unknown
	unknown_0
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315557G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
Л
V
)__inference_restored_function_body_315428
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310355^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
Л
V
)__inference_restored_function_body_315544
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310611^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
√
Х
(__inference_dense_5_layer_call_fn_315263

inputs
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_314777o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Щ
-
__inference__destroyer_315614
identityы
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315610G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Щ
-
__inference__destroyer_315576
identityы
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315572G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
«
Ч
(__inference_dense_3_layer_call_fn_315223

inputs
unknown:	И
	unknown_0:	И
identityИҐStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_314743p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€И`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ї
P
#__inference__update_step_xla_315135
gradient
variable:	И*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	И: *
	_noinline(:I E

_output_shapes
:	И
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
љ
h
__inference__initializer_315603
unknown
	unknown_0
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315595G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
з
r
)__inference_restored_function_body_315518
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_310623^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
Ы
-
__inference__destroyer_310567
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
™
H
__inference__creator_315470
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315467^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
т!
Н
$__inference_signature_wrapper_310563

inputs
inputs_1
	inputs_10
inputs_2
inputs_3	
inputs_4	
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5	
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9	

unknown_10	

unknown_11	

unknown_12

unknown_13	

unknown_14	

unknown_15	

unknown_16	

unknown_17

unknown_18	

unknown_19	
identity

identity_1

identity_2

identity_3

identity_4	

identity_5

identity_6ИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*+
Tin$
"2 														*
Tout
	2	*Н
_output_shapes{
y:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *"
fR
__inference_pruned_310515`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€m

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:€€€€€€€€€m

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*Р
_input_shapesю
ы:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_10:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_7:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_8:Q
M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_9:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
√(
√
C__inference_model_1_layer_call_and_return_conditional_losses_315092
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_59
&dense_3_matmul_readvariableop_resource:	И6
'dense_3_biasadd_readvariableop_resource:	И9
&dense_4_matmul_readvariableop_resource:	И5
'dense_4_biasadd_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:
identityИҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOp[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :У
concatenate_4/concatConcatV2inputs_0inputs_1"concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Э
concatenate_5/concatConcatV2inputs_2inputs_3inputs_4"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€a
concatenate_6/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :c
concatenate_6/concat/concatIdentityinputs_5*
T0*'
_output_shapes
:€€€€€€€€€[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :г
concatenate_7/concatConcatV2concatenate_4/concat:output:0concatenate_5/concat:output:0$concatenate_6/concat/concat:output:0"concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype0С
dense_3/MatMulMatMulconcatenate_7/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ИГ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype0П
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Иa
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ИЕ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype0Н
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Н
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Й
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5
Ы
-
__inference__destroyer_310325
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Ґ#
Б
C__inference_model_1_layer_call_and_return_conditional_losses_314980
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder!
dense_3_314964:	И
dense_3_314966:	И!
dense_4_314969:	И
dense_4_314971: 
dense_5_314974:
dense_5_314976:
identityИҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallг
concatenate_4/PartitionedCallPartitionedCallbuilding_type_xfinsured_period_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_314702ч
concatenate_5/PartitionedCallPartitionedCallbuilding_fenced_xfbuilding_painted_xfsettlement_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_314712 
concatenate_6/PartitionedCallPartitionedCallplaceholder*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_314720Ј
concatenate_7/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0&concatenate_5/PartitionedCall:output:0&concatenate_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_314730Р
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_3_314964dense_3_314966*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_314743С
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_314969dense_4_314971*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_314760С
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_314974dense_5_314976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_314777w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ђ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:Y U
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:€€€€€€€€€
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameBuilding Dimension_xf
Э-
Ю
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_314475

inputs
inputs_1
inputs_2
inputs_3	
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5	
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9	

unknown_10	

unknown_11	

unknown_12

unknown_13	

unknown_14	

unknown_15	

unknown_16	

unknown_17

unknown_18	

unknown_19	
identity

identity_1

identity_2

identity_3

identity_4

identity_5ИҐStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask=
Shape_1Shapeinputs*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:џ
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :u
zeros/packedPackstrided_slice_1:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:M
zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Ф
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€о
StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3PlaceholderWithDefault:output:0inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*+
Tin$
"2 														*
Tout
	2	*Н
_output_shapes{
y:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *"
fR
__inference_pruned_310515k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€m

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:€€€€€€€€€m

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*э
_input_shapesл
и:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O	K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ы
-
__inference__destroyer_310571
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
∞
L
.__inference_concatenate_6_layer_call_fn_315193
inputs_0
identityє
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_314720`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0
Щ
-
__inference__destroyer_315499
identityы
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315495G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
™
H
__inference__creator_315508
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315505^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
Л
V
)__inference_restored_function_body_315621
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310601^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
з
r
)__inference_restored_function_body_315441
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_310617^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
ђ
K
#__inference__update_step_xla_315160
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
®
√
__inference__initializer_310311!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityИҐ,text_file_init/InitializeTableFromTextFileV2у
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexю€€€€€€€€*
value_index€€€€€€€€€G
ConstConst*
_output_shapes
: *
dtype0*
value	B :u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
®
;
__inference__creator_310606
identityИҐ
hash_tableƒ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*ѕ
shared_nameњЉhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/43/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_1_vocabulary', shape=(), dtype=string)_-2_-1_load_310305_310602*
use_node_name_sharing(*
value_dtype0	S
NoOpNoOp^hash_table*"
_acd_function_control_output(*
_output_shapes
 W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
Ј
Б
I__inference_concatenate_7_layer_call_and_return_conditional_losses_314730

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Щ
-
__inference__destroyer_315537
identityы
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315533G
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
з
r
)__inference_restored_function_body_315595
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_310596^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
ѓ
L
#__inference__update_step_xla_315140
gradient
variable:	И*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:И: *
	_noinline(:E A

_output_shapes	
:И
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
™
H
__inference__creator_315624
identityИҐStatefulPartitionedCallЛ
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315621^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
ї
9
)__inference_restored_function_body_315572
identity–
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__destroyer_310321O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
¬
Г
I__inference_concatenate_7_layer_call_and_return_conditional_losses_315214
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2
й
Ќ
$__inference_signature_wrapper_314341
examples
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5	
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9	

unknown_10	

unknown_11	

unknown_12

unknown_13	

unknown_14	

unknown_15	

unknown_16	

unknown_17

unknown_18	

unknown_19	

unknown_20:	И

unknown_21:	И

unknown_22:	И

unknown_23:

unknown_24:

unknown_25:
identityИҐStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallexamplesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*'
Tin 
2												*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *0
f+R)
'__inference_serve_tf_examples_fn_310907o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
examples:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ж#
щ
9__inference_transform_features_layer_layer_call_fn_315340
inputs_building_dimension
inputs_building_fenced
inputs_building_painted
inputs_building_type	
inputs_customer_id
inputs_garden
inputs_geo_code
inputs_insured_period
inputs_numberofwindows
inputs_settlement
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5	
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9	

unknown_10	

unknown_11	

unknown_12

unknown_13	

unknown_14	

unknown_15	

unknown_16	

unknown_17

unknown_18	

unknown_19	
identity

identity_1

identity_2

identity_3

identity_4

identity_5ИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputs_building_dimensioninputs_building_fencedinputs_building_paintedinputs_building_typeinputs_customer_idinputs_gardeninputs_geo_codeinputs_insured_periodinputs_numberofwindowsinputs_settlementunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19**
Tin#
!2													*
Tout

2*
_collective_manager_ids
 *z
_output_shapesh
f:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_314475k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€m

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:€€€€€€€€€m

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*э
_input_shapesл
и:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:€€€€€€€€€
3
_user_specified_nameinputs_building_dimension:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_nameinputs_building_fenced:`\
'
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs_building_painted:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_nameinputs_building_type:[W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameinputs_customer_id:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinputs_garden:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_geo_code:^Z
'
_output_shapes
:€€€€€€€€€
/
_user_specified_nameinputs_insured_period:_[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_nameinputs_numberofwindows:Z	V
'
_output_shapes
:€€€€€€€€€
+
_user_specified_nameinputs_settlement:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
∆
Ц
(__inference_dense_4_layer_call_fn_315243

inputs
unknown:	И
	unknown_0:
identityИҐStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_314760o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€И: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€И
 
_user_specified_nameinputs
™!
Ѓ
9__inference_transform_features_layer_layer_call_fn_314530
placeholder
building_fenced
building_painted
building_type	
placeholder_1

garden
geo_code
insured_period
numberofwindows

settlement
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5	
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9	

unknown_10	

unknown_11	

unknown_12

unknown_13	

unknown_14	

unknown_15	

unknown_16	

unknown_17

unknown_18	

unknown_19	
identity

identity_1

identity_2

identity_3

identity_4

identity_5ИҐStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallplaceholderbuilding_fencedbuilding_paintedbuilding_typeplaceholder_1gardengeo_codeinsured_periodnumberofwindows
settlementunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19**
Tin#
!2													*
Tout

2*
_collective_manager_ids
 *z
_output_shapesh
f:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *]
fXRV
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_314475k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:€€€€€€€€€q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:€€€€€€€€€m

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:€€€€€€€€€m

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*#
_output_shapes
:€€€€€€€€€q

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*(
_construction_contextkEagerRuntime*э
_input_shapesл
и:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:€€€€€€€€€
,
_user_specified_nameBuilding Dimension:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameBuilding_Fenced:YU
'
_output_shapes
:€€€€€€€€€
*
_user_specified_nameBuilding_Painted:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_nameBuilding_Type:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_nameCustomer Id:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameGarden:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
Geo_Code:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_nameInsured_Period:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameNumberOfWindows:S	O
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
Settlement:


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
з
r
)__inference_restored_function_body_315557
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_310581^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
б
V
)__inference_restored_function_body_315705
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310606^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
®
√
__inference__initializer_310617!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityИҐ,text_file_init/InitializeTableFromTextFileV2у
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexю€€€€€€€€*
value_index€€€€€€€€€G
ConstConst*
_output_shapes
: *
dtype0*
value	B :u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
Ы
-
__inference__destroyer_310321
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
з
r
)__inference_restored_function_body_315634
unknown
	unknown_0
identityИҐStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__initializer_310317^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
б
V
)__inference_restored_function_body_315688
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCall*	
Tin
 *
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *$
fR
__inference__creator_310601^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*
_input_shapes 22
StatefulPartitionedCallStatefulPartitionedCall
®
√
__inference__initializer_310596!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityИҐ,text_file_init/InitializeTableFromTextFileV2у
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexю€€€€€€€€*
value_index€€€€€€€€€G
ConstConst*
_output_shapes
: *
dtype0*
value	B :u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
с
ћ
(__inference_model_1_layer_call_fn_315054
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown:	И
	unknown_0:	И
	unknown_1:	И
	unknown_2:
	unknown_3:
	unknown_4:
identityИҐStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_314915o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5
љ
h
__inference__initializer_315488
unknown
	unknown_0
identityИҐStatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *2
f-R+
)__inference_restored_function_body_315480G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 22
StatefulPartitionedCallStatefulPartitionedCall: 

_output_shapes
: 
√(
√
C__inference_model_1_layer_call_and_return_conditional_losses_315130
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_59
&dense_3_matmul_readvariableop_resource:	И6
'dense_3_biasadd_readvariableop_resource:	И9
&dense_4_matmul_readvariableop_resource:	И5
'dense_4_biasadd_readvariableop_resource:8
&dense_5_matmul_readvariableop_resource:5
'dense_5_biasadd_readvariableop_resource:
identityИҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOp[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :У
concatenate_4/concatConcatV2inputs_0inputs_1"concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Э
concatenate_5/concatConcatV2inputs_2inputs_3inputs_4"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€a
concatenate_6/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :c
concatenate_6/concat/concatIdentityinputs_5*
T0*'
_output_shapes
:€€€€€€€€€[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :г
concatenate_7/concatConcatV2concatenate_4/concat:output:0concatenate_5/concat:output:0$concatenate_6/concat/concat:output:0"concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Е
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype0С
dense_3/MatMulMatMulconcatenate_7/concat:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ИГ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:И*
dtype0П
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Иa
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ИЕ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	И*
dtype0Н
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€`
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Н
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
dense_5/SigmoidSigmoiddense_5/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
IdentityIdentitydense_5/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Й
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5
®
√
__inference__initializer_310623!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityИҐ,text_file_init/InitializeTableFromTextFileV2у
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexю€€€€€€€€*
value_index€€€€€€€€€G
ConstConst*
_output_shapes
: *
dtype0*
value	B :u
NoOpNoOp-^text_file_init/InitializeTableFromTextFileV2*"
_acd_function_control_output(*
_output_shapes
 L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 2\
,text_file_init/InitializeTableFromTextFileV2,text_file_init/InitializeTableFromTextFileV2: 

_output_shapes
: 
ю!
—
C__inference_model_1_layer_call_and_return_conditional_losses_314915

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5!
dense_3_314899:	И
dense_3_314901:	И!
dense_4_314904:	И
dense_4_314906: 
dense_5_314909:
dense_5_314911:
identityИҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCall–
concatenate_4/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_314702Ё
concatenate_5/PartitionedCallPartitionedCallinputs_2inputs_3inputs_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_314712«
concatenate_6/PartitionedCallPartitionedCallinputs_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_314720Ј
concatenate_7/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0&concatenate_5/PartitionedCall:output:0&concatenate_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_314730Р
dense_3/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_3_314899dense_3_314901*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€И*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_314743С
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_314904dense_4_314906*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_314760С
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_314909dense_5_314911*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_314777w
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ђ
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Т
_input_shapesА
~:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
є
g
I__inference_concatenate_6_layer_call_and_return_conditional_losses_315199
inputs_0
identityS
concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :U
concat/concatIdentityinputs_0*
T0*'
_output_shapes
:€€€€€€€€€^
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0"Ж
N
saver_filename:0StatefulPartitionedCall_16:0StatefulPartitionedCall_178"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ђ
serving_defaultЧ
9
examples-
serving_default_examples:0€€€€€€€€€>
output_02
StatefulPartitionedCall_9:0€€€€€€€€€tensorflow/serving/predict2M

asset_path_initializer:0/vocab_compute_and_apply_vocabulary_2_vocabulary2O

asset_path_initializer_1:0/vocab_compute_and_apply_vocabulary_1_vocabulary2M

asset_path_initializer_2:0-vocab_compute_and_apply_vocabulary_vocabulary:РЃ
ъ
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-0
layer-10
layer_with_weights-1
layer-11
layer_with_weights-2
layer-12
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	tft_layer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
•
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
•
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
•
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
•
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
ї
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
ї
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias"
_tf_keras_layer
ї
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
Ћ
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses
$N _saved_model_loader_tracked_dict"
_tf_keras_model
J
60
71
>2
?3
F4
G5"
trackable_list_wrapper
J
60
71
>2
?3
F4
G5"
trackable_list_wrapper
 "
trackable_list_wrapper
 
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
’
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_32к
(__inference_model_1_layer_call_fn_314799
(__inference_model_1_layer_call_fn_315032
(__inference_model_1_layer_call_fn_315054
(__inference_model_1_layer_call_fn_314952њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zTtrace_0zUtrace_1zVtrace_2zWtrace_3
Ѕ
Xtrace_0
Ytrace_1
Ztrace_2
[trace_32÷
C__inference_model_1_layer_call_and_return_conditional_losses_315092
C__inference_model_1_layer_call_and_return_conditional_losses_315130
C__inference_model_1_layer_call_and_return_conditional_losses_314980
C__inference_model_1_layer_call_and_return_conditional_losses_315008њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zXtrace_0zYtrace_1zZtrace_2z[trace_3
ЈBі
!__inference__wrapped_model_314380Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ь
\
_variables
]_iterations
^_learning_rate
__index_dict
`
_momentums
a_velocities
b_update_step_xla"
experimentalOptimizer
,
cserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
т
itrace_02’
.__inference_concatenate_4_layer_call_fn_315166Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zitrace_0
Н
jtrace_02р
I__inference_concatenate_4_layer_call_and_return_conditional_losses_315173Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zjtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
т
ptrace_02’
.__inference_concatenate_5_layer_call_fn_315180Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zptrace_0
Н
qtrace_02р
I__inference_concatenate_5_layer_call_and_return_conditional_losses_315188Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zqtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
т
wtrace_02’
.__inference_concatenate_6_layer_call_fn_315193Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zwtrace_0
Н
xtrace_02р
I__inference_concatenate_6_layer_call_and_return_conditional_losses_315199Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zxtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
т
~trace_02’
.__inference_concatenate_7_layer_call_fn_315206Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z~trace_0
Н
trace_02р
I__inference_concatenate_7_layer_call_and_return_conditional_losses_315214Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ztrace_0
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
о
Еtrace_02ѕ
(__inference_dense_3_layer_call_fn_315223Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЕtrace_0
Й
Жtrace_02к
C__inference_dense_3_layer_call_and_return_conditional_losses_315234Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЖtrace_0
!:	И2dense_3/kernel
:И2dense_3/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
о
Мtrace_02ѕ
(__inference_dense_4_layer_call_fn_315243Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zМtrace_0
Й
Нtrace_02к
C__inference_dense_4_layer_call_and_return_conditional_losses_315254Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zНtrace_0
!:	И2dense_4/kernel
:2dense_4/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Оnon_trainable_variables
Пlayers
Рmetrics
 Сlayer_regularization_losses
Тlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
о
Уtrace_02ѕ
(__inference_dense_5_layer_call_fn_315263Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zУtrace_0
Й
Фtrace_02к
C__inference_dense_5_layer_call_and_return_conditional_losses_315274Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zФtrace_0
 :2dense_5/kernel
:2dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Хnon_trainable_variables
Цlayers
Чmetrics
 Шlayer_regularization_losses
Щlayer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
÷
Ъtrace_0
Ыtrace_12Ы
9__inference_transform_features_layer_layer_call_fn_314530
9__inference_transform_features_layer_layer_call_fn_315340Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЪtrace_0zЫtrace_1
М
Ьtrace_0
Эtrace_12—
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_315422
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_314678Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЬtrace_0zЭtrace_1
Ч
Ю	_imported
Я_wrapped_function
†_structured_inputs
°_structured_outputs
Ґ_output_to_inputs_map"
trackable_dict_wrapper
 "
trackable_list_wrapper
Ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
0
£0
§1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
еBв
(__inference_model_1_layer_call_fn_314799Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠B™
(__inference_model_1_layer_call_fn_315032inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≠B™
(__inference_model_1_layer_call_fn_315054inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
еBв
(__inference_model_1_layer_call_fn_314952Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»B≈
C__inference_model_1_layer_call_and_return_conditional_losses_315092inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
»B≈
C__inference_model_1_layer_call_and_return_conditional_losses_315130inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
C__inference_model_1_layer_call_and_return_conditional_losses_314980Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
АBэ
C__inference_model_1_layer_call_and_return_conditional_losses_315008Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
К
]0
•1
¶2
І3
®4
©5
™6
Ђ7
ђ8
≠9
Ѓ10
ѓ11
∞12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
P
•0
І1
©2
Ђ3
≠4
ѓ5"
trackable_list_wrapper
P
¶0
®1
™2
ђ3
Ѓ4
∞5"
trackable_list_wrapper
≈
±trace_0
≤trace_1
≥trace_2
іtrace_3
µtrace_4
ґtrace_52Ъ
#__inference__update_step_xla_315135
#__inference__update_step_xla_315140
#__inference__update_step_xla_315145
#__inference__update_step_xla_315150
#__inference__update_step_xla_315155
#__inference__update_step_xla_315160є
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0z±trace_0z≤trace_1z≥trace_2zіtrace_3zµtrace_4zґtrace_5
Ю
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20B…
$__inference_signature_wrapper_314341examples"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈ	capture_0zЄ	capture_1zє	capture_2zЇ	capture_3zї	capture_4zЉ	capture_5zљ	capture_6zЊ	capture_7zњ	capture_9zј
capture_10zЅ
capture_11z¬
capture_12z√
capture_14zƒ
capture_15z≈
capture_16z∆
capture_17z«
capture_19z»
capture_20
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
.__inference_concatenate_4_layer_call_fn_315166inputs_0inputs_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЙBЖ
I__inference_concatenate_4_layer_call_and_return_conditional_losses_315173inputs_0inputs_1"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
шBх
.__inference_concatenate_5_layer_call_fn_315180inputs_0inputs_1inputs_2"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
УBР
I__inference_concatenate_5_layer_call_and_return_conditional_losses_315188inputs_0inputs_1inputs_2"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
.__inference_concatenate_6_layer_call_fn_315193inputs_0"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
€Bь
I__inference_concatenate_6_layer_call_and_return_conditional_losses_315199inputs_0"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
шBх
.__inference_concatenate_7_layer_call_fn_315206inputs_0inputs_1inputs_2"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
УBР
I__inference_concatenate_7_layer_call_and_return_conditional_losses_315214inputs_0inputs_1inputs_2"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_3_layer_call_fn_315223inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_3_layer_call_and_return_conditional_losses_315234inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_4_layer_call_fn_315243inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_4_layer_call_and_return_conditional_losses_315254inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_5_layer_call_fn_315263inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_5_layer_call_and_return_conditional_losses_315274inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
…
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20Bф
9__inference_transform_features_layer_layer_call_fn_314530Building DimensionBuilding_FencedBuilding_PaintedBuilding_TypeCustomer IdGardenGeo_CodeInsured_PeriodNumberOfWindows
Settlement
"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈ	capture_0zЄ	capture_1zє	capture_2zЇ	capture_3zї	capture_4zЉ	capture_5zљ	capture_6zЊ	capture_7zњ	capture_9zј
capture_10zЅ
capture_11z¬
capture_12z√
capture_14zƒ
capture_15z≈
capture_16z∆
capture_17z«
capture_19z»
capture_20
П
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20BЇ
9__inference_transform_features_layer_layer_call_fn_315340inputs_building_dimensioninputs_building_fencedinputs_building_paintedinputs_building_typeinputs_customer_idinputs_gardeninputs_geo_codeinputs_insured_periodinputs_numberofwindowsinputs_settlement
"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈ	capture_0zЄ	capture_1zє	capture_2zЇ	capture_3zї	capture_4zЉ	capture_5zљ	capture_6zЊ	capture_7zњ	capture_9zј
capture_10zЅ
capture_11z¬
capture_12z√
capture_14zƒ
capture_15z≈
capture_16z∆
capture_17z«
capture_19z»
capture_20
™
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20B’
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_315422inputs_building_dimensioninputs_building_fencedinputs_building_paintedinputs_building_typeinputs_customer_idinputs_gardeninputs_geo_codeinputs_insured_periodinputs_numberofwindowsinputs_settlement
"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈ	capture_0zЄ	capture_1zє	capture_2zЇ	capture_3zї	capture_4zЉ	capture_5zљ	capture_6zЊ	capture_7zњ	capture_9zј
capture_10zЅ
capture_11z¬
capture_12z√
capture_14zƒ
capture_15z≈
capture_16z∆
capture_17z«
capture_19z»
capture_20
д
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20BП
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_314678Building DimensionBuilding_FencedBuilding_PaintedBuilding_TypeCustomer IdGardenGeo_CodeInsured_PeriodNumberOfWindows
Settlement
"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈ	capture_0zЄ	capture_1zє	capture_2zЇ	capture_3zї	capture_4zЉ	capture_5zљ	capture_6zЊ	capture_7zњ	capture_9zј
capture_10zЅ
capture_11z¬
capture_12z√
capture_14zƒ
capture_15z≈
capture_16z∆
capture_17z«
capture_19z»
capture_20
»
…created_variables
 	resources
Ћtrackable_objects
ћinitializers
Ќassets
ќ
signatures
$ѕ_self_saveable_object_factories
Яtransform_fn"
_generic_user_object
я
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20BК
__inference_pruned_310515inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10zЈ	capture_0zЄ	capture_1zє	capture_2zЇ	capture_3zї	capture_4zЉ	capture_5zљ	capture_6zЊ	capture_7zњ	capture_9zј
capture_10zЅ
capture_11z¬
capture_12z√
capture_14zƒ
capture_15z≈
capture_16z∆
capture_17z«
capture_19z»
capture_20
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
R
–	variables
—	keras_api

“total

”count"
_tf_keras_metric
c
‘	variables
’	keras_api

÷total

„count
Ў
_fn_kwargs"
_tf_keras_metric
&:$	И2Adam/m/dense_3/kernel
&:$	И2Adam/v/dense_3/kernel
 :И2Adam/m/dense_3/bias
 :И2Adam/v/dense_3/bias
&:$	И2Adam/m/dense_4/kernel
&:$	И2Adam/v/dense_4/kernel
:2Adam/m/dense_4/bias
:2Adam/v/dense_4/bias
%:#2Adam/m/dense_5/kernel
%:#2Adam/v/dense_5/kernel
:2Adam/m/dense_5/bias
:2Adam/v/dense_5/bias
шBх
#__inference__update_step_xla_315135gradientvariable"Ј
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
#__inference__update_step_xla_315140gradientvariable"Ј
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
#__inference__update_step_xla_315145gradientvariable"Ј
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
#__inference__update_step_xla_315150gradientvariable"Ј
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
#__inference__update_step_xla_315155gradientvariable"Ј
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
#__inference__update_step_xla_315160gradientvariable"Ј
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
!J	
Const_8jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstant
"J

Const_17jtf.TrackableConstant
"J

Const_16jtf.TrackableConstant
"J

Const_15jtf.TrackableConstant
"J

Const_14jtf.TrackableConstant
"J

Const_13jtf.TrackableConstant
"J

Const_12jtf.TrackableConstant
"J

Const_11jtf.TrackableConstant
"J

Const_10jtf.TrackableConstant
!J	
Const_9jtf.TrackableConstant
 "
trackable_list_wrapper
P
ў0
Џ1
џ2
№3
Ё4
ё5"
trackable_list_wrapper
 "
trackable_list_wrapper
8
я0
а1
б2"
trackable_list_wrapper
8
в0
г1
д2"
trackable_list_wrapper
-
еserving_default"
signature_map
 "
trackable_dict_wrapper
0
“0
”1"
trackable_list_wrapper
.
–	variables"
_generic_user_object
:  (2total
:  (2count
0
÷0
„1"
trackable_list_wrapper
.
‘	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
V
я_initializer
ж_create_resource
з_initialize
и_destroy_resourceR 
V
я_initializer
й_create_resource
к_initialize
л_destroy_resourceR 
V
а_initializer
м_create_resource
н_initialize
о_destroy_resourceR 
V
а_initializer
п_create_resource
р_initialize
с_destroy_resourceR 
V
б_initializer
т_create_resource
у_initialize
ф_destroy_resourceR 
V
б_initializer
х_create_resource
ц_initialize
ч_destroy_resourceR 
T
в	_filename
$ш_self_saveable_object_factories"
_generic_user_object
T
г	_filename
$щ_self_saveable_object_factories"
_generic_user_object
T
д	_filename
$ъ_self_saveable_object_factories"
_generic_user_object
*
*
* 
€
Ј	capture_0
Є	capture_1
є	capture_2
Ї	capture_3
ї	capture_4
Љ	capture_5
љ	capture_6
Њ	capture_7
њ	capture_9
ј
capture_10
Ѕ
capture_11
¬
capture_12
√
capture_14
ƒ
capture_15
≈
capture_16
∆
capture_17
«
capture_19
»
capture_20B™
$__inference_signature_wrapper_310563inputsinputs_1	inputs_10inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈ	capture_0zЄ	capture_1zє	capture_2zЇ	capture_3zї	capture_4zЉ	capture_5zљ	capture_6zЊ	capture_7zњ	capture_9zј
capture_10zЅ
capture_11z¬
capture_12z√
capture_14zƒ
capture_15z≈
capture_16z∆
capture_17z«
capture_19z»
capture_20
ќ
ыtrace_02ѓ
__inference__creator_315431П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zыtrace_0
“
ьtrace_02≥
__inference__initializer_315449П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zьtrace_0
–
эtrace_02±
__inference__destroyer_315460П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zэtrace_0
ќ
юtrace_02ѓ
__inference__creator_315470П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zюtrace_0
“
€trace_02≥
__inference__initializer_315488П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z€trace_0
–
Аtrace_02±
__inference__destroyer_315499П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zАtrace_0
ќ
Бtrace_02ѓ
__inference__creator_315508П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zБtrace_0
“
Вtrace_02≥
__inference__initializer_315526П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zВtrace_0
–
Гtrace_02±
__inference__destroyer_315537П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zГtrace_0
ќ
Дtrace_02ѓ
__inference__creator_315547П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zДtrace_0
“
Еtrace_02≥
__inference__initializer_315565П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЕtrace_0
–
Жtrace_02±
__inference__destroyer_315576П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЖtrace_0
ќ
Зtrace_02ѓ
__inference__creator_315585П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЗtrace_0
“
Иtrace_02≥
__inference__initializer_315603П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zИtrace_0
–
Йtrace_02±
__inference__destroyer_315614П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЙtrace_0
ќ
Кtrace_02ѓ
__inference__creator_315624П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zКtrace_0
“
Лtrace_02≥
__inference__initializer_315642П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЛtrace_0
–
Мtrace_02±
__inference__destroyer_315653П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zМtrace_0
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
≤Bѓ
__inference__creator_315431"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
÷
в	capture_0B≥
__inference__initializer_315449"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zв	capture_0
іB±
__inference__destroyer_315460"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference__creator_315470"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
÷
в	capture_0B≥
__inference__initializer_315488"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zв	capture_0
іB±
__inference__destroyer_315499"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference__creator_315508"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
÷
г	capture_0B≥
__inference__initializer_315526"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zг	capture_0
іB±
__inference__destroyer_315537"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference__creator_315547"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
÷
г	capture_0B≥
__inference__initializer_315565"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zг	capture_0
іB±
__inference__destroyer_315576"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference__creator_315585"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
÷
д	capture_0B≥
__inference__initializer_315603"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zд	capture_0
іB±
__inference__destroyer_315614"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≤Bѓ
__inference__creator_315624"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
÷
д	capture_0B≥
__inference__initializer_315642"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zд	capture_0
іB±
__inference__destroyer_315653"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ @
__inference__creator_315431!Ґ

Ґ 
™ "К
unknown @
__inference__creator_315470!Ґ

Ґ 
™ "К
unknown @
__inference__creator_315508!Ґ

Ґ 
™ "К
unknown @
__inference__creator_315547!Ґ

Ґ 
™ "К
unknown @
__inference__creator_315585!Ґ

Ґ 
™ "К
unknown @
__inference__creator_315624!Ґ

Ґ 
™ "К
unknown B
__inference__destroyer_315460!Ґ

Ґ 
™ "К
unknown B
__inference__destroyer_315499!Ґ

Ґ 
™ "К
unknown B
__inference__destroyer_315537!Ґ

Ґ 
™ "К
unknown B
__inference__destroyer_315576!Ґ

Ґ 
™ "К
unknown B
__inference__destroyer_315614!Ґ

Ґ 
™ "К
unknown B
__inference__destroyer_315653!Ґ

Ґ 
™ "К
unknown J
__inference__initializer_315449'вўҐ

Ґ 
™ "К
unknown J
__inference__initializer_315488'вўҐ

Ґ 
™ "К
unknown J
__inference__initializer_315526'гџҐ

Ґ 
™ "К
unknown J
__inference__initializer_315565'гџҐ

Ґ 
™ "К
unknown J
__inference__initializer_315603'дЁҐ

Ґ 
™ "К
unknown J
__inference__initializer_315642'дЁҐ

Ґ 
™ "К
unknown Ч
#__inference__update_step_xla_315135pjҐg
`Ґ]
К
gradient	И
5Т2	Ґ
ъ	И
А
p
` VariableSpec 
`†ИЭ¶Щк?
™ "
 П
#__inference__update_step_xla_315140hbҐ_
XҐU
К
gradientИ
1Т.	Ґ
ъИ
А
p
` VariableSpec 
`аўі≠Щк?
™ "
 Ч
#__inference__update_step_xla_315145pjҐg
`Ґ]
К
gradient	И
5Т2	Ґ
ъ	И
А
p
` VariableSpec 
`†∞¬ЫЩк?
™ "
 Н
#__inference__update_step_xla_315150f`Ґ]
VҐS
К
gradient
0Т-	Ґ
ъ
А
p
` VariableSpec 
`А†¬ЫЩк?
™ "
 Х
#__inference__update_step_xla_315155nhҐe
^Ґ[
К
gradient
4Т1	Ґ
ъ
А
p
` VariableSpec 
`†НФХЩк?
™ "
 Н
#__inference__update_step_xla_315160f`Ґ]
VҐS
К
gradient
0Т-	Ґ
ъ
А
p
` VariableSpec 
`ач№БЦк?
™ "
 Л
!__inference__wrapped_model_314380е67>?FGІҐ£
ЫҐЧ
ФЪР
*К'
Building_Type_xf€€€€€€€€€
+К(
Insured_Period_xf€€€€€€€€€
,К)
Building_Fenced_xf€€€€€€€€€
-К*
Building_Painted_xf€€€€€€€€€
'К$
Settlement_xf€€€€€€€€€
/К,
Building Dimension_xf€€€€€€€€€
™ "1™.
,
dense_5!К
dense_5€€€€€€€€€Ў
I__inference_concatenate_4_layer_call_and_return_conditional_losses_315173КZҐW
PҐM
KЪH
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ±
.__inference_concatenate_4_layer_call_fn_315166ZҐW
PҐM
KЪH
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
™ "!К
unknown€€€€€€€€€ь
I__inference_concatenate_5_layer_call_and_return_conditional_losses_315188Ѓ~Ґ{
tҐq
oЪl
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ÷
.__inference_concatenate_5_layer_call_fn_315180£~Ґ{
tҐq
oЪl
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
™ "!К
unknown€€€€€€€€€≥
I__inference_concatenate_6_layer_call_and_return_conditional_losses_315199f6Ґ3
,Ґ)
'Ъ$
"К
inputs_0€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Н
.__inference_concatenate_6_layer_call_fn_315193[6Ґ3
,Ґ)
'Ъ$
"К
inputs_0€€€€€€€€€
™ "!К
unknown€€€€€€€€€ь
I__inference_concatenate_7_layer_call_and_return_conditional_losses_315214Ѓ~Ґ{
tҐq
oЪl
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ÷
.__inference_concatenate_7_layer_call_fn_315206£~Ґ{
tҐq
oЪl
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ђ
C__inference_dense_3_layer_call_and_return_conditional_losses_315234d67/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "-Ґ*
#К 
tensor_0€€€€€€€€€И
Ъ Е
(__inference_dense_3_layer_call_fn_315223Y67/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ""К
unknown€€€€€€€€€ИЂ
C__inference_dense_4_layer_call_and_return_conditional_losses_315254d>?0Ґ-
&Ґ#
!К
inputs€€€€€€€€€И
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Е
(__inference_dense_4_layer_call_fn_315243Y>?0Ґ-
&Ґ#
!К
inputs€€€€€€€€€И
™ "!К
unknown€€€€€€€€€™
C__inference_dense_5_layer_call_and_return_conditional_losses_315274cFG/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Д
(__inference_dense_5_layer_call_fn_315263XFG/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€∞
C__inference_model_1_layer_call_and_return_conditional_losses_314980и67>?FGѓҐЂ
£ҐЯ
ФЪР
*К'
Building_Type_xf€€€€€€€€€
+К(
Insured_Period_xf€€€€€€€€€
,К)
Building_Fenced_xf€€€€€€€€€
-К*
Building_Painted_xf€€€€€€€€€
'К$
Settlement_xf€€€€€€€€€
/К,
Building Dimension_xf€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ∞
C__inference_model_1_layer_call_and_return_conditional_losses_315008и67>?FGѓҐЂ
£ҐЯ
ФЪР
*К'
Building_Type_xf€€€€€€€€€
+К(
Insured_Period_xf€€€€€€€€€
,К)
Building_Fenced_xf€€€€€€€€€
-К*
Building_Painted_xf€€€€€€€€€
'К$
Settlement_xf€€€€€€€€€
/К,
Building Dimension_xf€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ш
C__inference_model_1_layer_call_and_return_conditional_losses_315092∞67>?FGчҐу
лҐз
№ЪЎ
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
"К
inputs_5€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ ш
C__inference_model_1_layer_call_and_return_conditional_losses_315130∞67>?FGчҐу
лҐз
№ЪЎ
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
"К
inputs_5€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ К
(__inference_model_1_layer_call_fn_314799Ё67>?FGѓҐЂ
£ҐЯ
ФЪР
*К'
Building_Type_xf€€€€€€€€€
+К(
Insured_Period_xf€€€€€€€€€
,К)
Building_Fenced_xf€€€€€€€€€
-К*
Building_Painted_xf€€€€€€€€€
'К$
Settlement_xf€€€€€€€€€
/К,
Building Dimension_xf€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€К
(__inference_model_1_layer_call_fn_314952Ё67>?FGѓҐЂ
£ҐЯ
ФЪР
*К'
Building_Type_xf€€€€€€€€€
+К(
Insured_Period_xf€€€€€€€€€
,К)
Building_Fenced_xf€€€€€€€€€
-К*
Building_Painted_xf€€€€€€€€€
'К$
Settlement_xf€€€€€€€€€
/К,
Building Dimension_xf€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€“
(__inference_model_1_layer_call_fn_315032•67>?FGчҐу
лҐз
№ЪЎ
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
"К
inputs_5€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€“
(__inference_model_1_layer_call_fn_315054•67>?FGчҐу
лҐз
№ЪЎ
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
"К
inputs_5€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€ћ	
__inference_pruned_310515Ѓ	*ЈЄєЇїЉљЊўњјЅ¬џ√ƒ≈∆Ё«» Ґ∆
ЊҐЇ
Ј™≥
I
Building Dimension3К0
inputs_building_dimension€€€€€€€€€
C
Building_Fenced0К-
inputs_building_fenced€€€€€€€€€
E
Building_Painted1К.
inputs_building_painted€€€€€€€€€
?
Building_Type.К+
inputs_building_type€€€€€€€€€	
/
Claim&К#
inputs_claim€€€€€€€€€	
;
Customer Id,К)
inputs_customer_id€€€€€€€€€
1
Garden'К$
inputs_garden€€€€€€€€€
5
Geo_Code)К&
inputs_geo_code€€€€€€€€€
A
Insured_Period/К,
inputs_insured_period€€€€€€€€€
C
NumberOfWindows0К-
inputs_numberofwindows€€€€€€€€€
9

Settlement+К(
inputs_settlement€€€€€€€€€
™ "≤™Ѓ
D
Building Dimension_xf+К(
building_dimension_xf€€€€€€€€€
B
Building_Fenced_xf,К)
building_fenced_xf€€€€€€€€€
D
Building_Painted_xf-К*
building_painted_xf€€€€€€€€€
:
Building_Type_xf&К#
building_type_xf€€€€€€€€€
(
ClaimК
claim€€€€€€€€€	
<
Insured_Period_xf'К$
insured_period_xf€€€€€€€€€
8
Settlement_xf'К$
settlement_xf€€€€€€€€€Ђ
$__inference_signature_wrapper_310563В*ЈЄєЇїЉљЊўњјЅ¬џ√ƒ≈∆Ё«»ЮҐЪ
Ґ 
Т™О
*
inputs К
inputs€€€€€€€€€
.
inputs_1"К
inputs_1€€€€€€€€€
0
	inputs_10#К 
	inputs_10€€€€€€€€€
.
inputs_2"К
inputs_2€€€€€€€€€
.
inputs_3"К
inputs_3€€€€€€€€€	
.
inputs_4"К
inputs_4€€€€€€€€€	
.
inputs_5"К
inputs_5€€€€€€€€€
.
inputs_6"К
inputs_6€€€€€€€€€
.
inputs_7"К
inputs_7€€€€€€€€€
.
inputs_8"К
inputs_8€€€€€€€€€
.
inputs_9"К
inputs_9€€€€€€€€€"≤™Ѓ
D
Building Dimension_xf+К(
building_dimension_xf€€€€€€€€€
B
Building_Fenced_xf,К)
building_fenced_xf€€€€€€€€€
D
Building_Painted_xf-К*
building_painted_xf€€€€€€€€€
:
Building_Type_xf&К#
building_type_xf€€€€€€€€€
(
ClaimК
claim€€€€€€€€€	
<
Insured_Period_xf'К$
insured_period_xf€€€€€€€€€
8
Settlement_xf'К$
settlement_xf€€€€€€€€€Ћ
$__inference_signature_wrapper_314341Ґ0ЈЄєЇїЉљЊўњјЅ¬џ√ƒ≈∆Ё«»67>?FG9Ґ6
Ґ 
/™,
*
examplesК
examples€€€€€€€€€"3™0
.
output_0"К
output_0€€€€€€€€€®	
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_314678ѕ*ЈЄєЇїЉљЊўњјЅ¬џ√ƒ≈∆Ё«»”Ґѕ
«Ґ√
ј™Љ
B
Building Dimension,К)
Building Dimension€€€€€€€€€
<
Building_Fenced)К&
Building_Fenced€€€€€€€€€
>
Building_Painted*К'
Building_Painted€€€€€€€€€
8
Building_Type'К$
Building_Type€€€€€€€€€	
4
Customer Id%К"
Customer Id€€€€€€€€€
*
Garden К
Garden€€€€€€€€€
.
Geo_Code"К
Geo_Code€€€€€€€€€
:
Insured_Period(К%
Insured_Period€€€€€€€€€
<
NumberOfWindows)К&
NumberOfWindows€€€€€€€€€
2

Settlement$К!

Settlement€€€€€€€€€
™ " Ґ∆
Њ™Ї
M
Building Dimension_xf4К1
tensor_0_building_dimension_xf€€€€€€€€€
K
Building_Fenced_xf5К2
tensor_0_building_fenced_xf€€€€€€€€€
M
Building_Painted_xf6К3
tensor_0_building_painted_xf€€€€€€€€€
C
Building_Type_xf/К,
tensor_0_building_type_xf€€€€€€€€€
E
Insured_Period_xf0К-
tensor_0_insured_period_xf€€€€€€€€€
A
Settlement_xf0К-
tensor_0_settlement_xf€€€€€€€€€
Ъ о	
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_315422Х	*ЈЄєЇїЉљЊўњјЅ¬џ√ƒ≈∆Ё«»ЩҐХ
НҐЙ
Ж™В
I
Building Dimension3К0
inputs_building_dimension€€€€€€€€€
C
Building_Fenced0К-
inputs_building_fenced€€€€€€€€€
E
Building_Painted1К.
inputs_building_painted€€€€€€€€€
?
Building_Type.К+
inputs_building_type€€€€€€€€€	
;
Customer Id,К)
inputs_customer_id€€€€€€€€€
1
Garden'К$
inputs_garden€€€€€€€€€
5
Geo_Code)К&
inputs_geo_code€€€€€€€€€
A
Insured_Period/К,
inputs_insured_period€€€€€€€€€
C
NumberOfWindows0К-
inputs_numberofwindows€€€€€€€€€
9

Settlement+К(
inputs_settlement€€€€€€€€€
™ " Ґ∆
Њ™Ї
M
Building Dimension_xf4К1
tensor_0_building_dimension_xf€€€€€€€€€
K
Building_Fenced_xf5К2
tensor_0_building_fenced_xf€€€€€€€€€
M
Building_Painted_xf6К3
tensor_0_building_painted_xf€€€€€€€€€
C
Building_Type_xf/К,
tensor_0_building_type_xf€€€€€€€€€
E
Insured_Period_xf0К-
tensor_0_insured_period_xf€€€€€€€€€
A
Settlement_xf0К-
tensor_0_settlement_xf€€€€€€€€€
Ъ Ћ
9__inference_transform_features_layer_layer_call_fn_314530Н*ЈЄєЇїЉљЊўњјЅ¬џ√ƒ≈∆Ё«»”Ґѕ
«Ґ√
ј™Љ
B
Building Dimension,К)
Building Dimension€€€€€€€€€
<
Building_Fenced)К&
Building_Fenced€€€€€€€€€
>
Building_Painted*К'
Building_Painted€€€€€€€€€
8
Building_Type'К$
Building_Type€€€€€€€€€	
4
Customer Id%К"
Customer Id€€€€€€€€€
*
Garden К
Garden€€€€€€€€€
.
Geo_Code"К
Geo_Code€€€€€€€€€
:
Insured_Period(К%
Insured_Period€€€€€€€€€
<
NumberOfWindows)К&
NumberOfWindows€€€€€€€€€
2

Settlement$К!

Settlement€€€€€€€€€
™ "И™Д
D
Building Dimension_xf+К(
building_dimension_xf€€€€€€€€€
B
Building_Fenced_xf,К)
building_fenced_xf€€€€€€€€€
D
Building_Painted_xf-К*
building_painted_xf€€€€€€€€€
:
Building_Type_xf&К#
building_type_xf€€€€€€€€€
<
Insured_Period_xf'К$
insured_period_xf€€€€€€€€€
8
Settlement_xf'К$
settlement_xf€€€€€€€€€С	
9__inference_transform_features_layer_layer_call_fn_315340”*ЈЄєЇїЉљЊўњјЅ¬џ√ƒ≈∆Ё«»ЩҐХ
НҐЙ
Ж™В
I
Building Dimension3К0
inputs_building_dimension€€€€€€€€€
C
Building_Fenced0К-
inputs_building_fenced€€€€€€€€€
E
Building_Painted1К.
inputs_building_painted€€€€€€€€€
?
Building_Type.К+
inputs_building_type€€€€€€€€€	
;
Customer Id,К)
inputs_customer_id€€€€€€€€€
1
Garden'К$
inputs_garden€€€€€€€€€
5
Geo_Code)К&
inputs_geo_code€€€€€€€€€
A
Insured_Period/К,
inputs_insured_period€€€€€€€€€
C
NumberOfWindows0К-
inputs_numberofwindows€€€€€€€€€
9

Settlement+К(
inputs_settlement€€€€€€€€€
™ "И™Д
D
Building Dimension_xf+К(
building_dimension_xf€€€€€€€€€
B
Building_Fenced_xf,К)
building_fenced_xf€€€€€€€€€
D
Building_Painted_xf-К*
building_painted_xf€€€€€€€€€
:
Building_Type_xf&К#
building_type_xf€€€€€€€€€
<
Insured_Period_xf'К$
insured_period_xf€€€€€€€€€
8
Settlement_xf'К$
settlement_xf€€€€€€€€€