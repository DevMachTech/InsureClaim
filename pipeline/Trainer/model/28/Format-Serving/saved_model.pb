ЗЁ
И&&
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
	summarizeint
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
incompatible_shape_errorbool(
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
Ё
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype
.
Identity

input"T
output"T"	
Ttype
м
InitializeTableFromTextFileV2
table_handle
filename"
	key_indexint(0ўџџџџџџџџ"
value_indexint(0ўџџџџџџџџ"+

vocab_sizeintџџџџџџџџџ(0џџџџџџџџџ"
	delimiterstring	"
offsetint 
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
Touttype
2
LookupTableSizeV2
table_handle
size	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
U
NotEqual
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisintџџџџџџџџџ"	
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

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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 
9
VarIsInitializedOp
resource
is_initialized

&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.11.02v2.11.0-rc2-17-gd5b57ca93e58ЉШ
W
asset_path_initializerPlaceholder*
_output_shapes
: *
dtype0*
shape: 

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


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

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


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

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
џџџџџџџџџ
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
 *гJ
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *;+шD
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  ?
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *   
L
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *  @
L
Const_8Const*
_output_shapes
: *
dtype0*
valueB
 *  П
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
џџџџџџџџџ
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
џџџџџџџџџ
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

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336616

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336616

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336622

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336627

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336627

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336633

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336638

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336638

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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336644
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
Adam/v/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_8/bias
w
'Adam/v/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_8/bias
w
'Adam/m/dense_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/bias*
_output_shapes
:*
dtype0

Adam/v/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*&
shared_nameAdam/v/dense_8/kernel

)Adam/v/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_8/kernel*
_output_shapes

:H*
dtype0

Adam/m/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*&
shared_nameAdam/m/dense_8/kernel

)Adam/m/dense_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_8/kernel*
_output_shapes

:H*
dtype0
~
Adam/v/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*$
shared_nameAdam/v/dense_7/bias
w
'Adam/v/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/bias*
_output_shapes
:H*
dtype0
~
Adam/m/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*$
shared_nameAdam/m/dense_7/bias
w
'Adam/m/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/bias*
_output_shapes
:H*
dtype0

Adam/v/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*&
shared_nameAdam/v/dense_7/kernel

)Adam/v/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_7/kernel*
_output_shapes

:H*
dtype0

Adam/m/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*&
shared_nameAdam/m/dense_7/kernel

)Adam/m/dense_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_7/kernel*
_output_shapes

:H*
dtype0
~
Adam/v/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/v/dense_6/bias
w
'Adam/v/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/bias*
_output_shapes
:*
dtype0
~
Adam/m/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/m/dense_6/bias
w
'Adam/m/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/bias*
_output_shapes
:*
dtype0

Adam/v/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/v/dense_6/kernel

)Adam/v/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_6/kernel*
_output_shapes

:*
dtype0

Adam/m/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/m/dense_6/kernel

)Adam/m/dense_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_6/kernel*
_output_shapes

:*
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
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
dtype0
x
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*
shared_namedense_8/kernel
q
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes

:H*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:H*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:H*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:H*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:*
dtype0
x
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_6/kernel
q
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes

:*
dtype0
s
serving_default_examplesPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Ќ
StatefulPartitionedCall_9StatefulPartitionedCallserving_default_examplesConst_8Const_7Const_6Const_5Const_4Const_3Const_2Const_1StatefulPartitionedCall_8ConstConst_17Const_16Const_15StatefulPartitionedCall_5Const_14Const_13Const_12Const_11StatefulPartitionedCall_2Const_10Const_9dense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias*'
Tin 
2												*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_335269
e
ReadVariableOpReadVariableOp
Variable_2^Variable_2/Assign*
_output_shapes
: *
dtype0
л
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
GPU2*0J 8 *(
f#R!
__inference__initializer_336377
g
ReadVariableOp_1ReadVariableOp
Variable_2^Variable_2/Assign*
_output_shapes
: *
dtype0
н
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
GPU2*0J 8 *(
f#R!
__inference__initializer_336416
g
ReadVariableOp_2ReadVariableOp
Variable_1^Variable_1/Assign*
_output_shapes
: *
dtype0
н
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
GPU2*0J 8 *(
f#R!
__inference__initializer_336454
g
ReadVariableOp_3ReadVariableOp
Variable_1^Variable_1/Assign*
_output_shapes
: *
dtype0
н
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
GPU2*0J 8 *(
f#R!
__inference__initializer_336493
c
ReadVariableOp_4ReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
н
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
GPU2*0J 8 *(
f#R!
__inference__initializer_336531
c
ReadVariableOp_5ReadVariableOpVariable^Variable/Assign*
_output_shapes
: *
dtype0
н
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
GPU2*0J 8 *(
f#R!
__inference__initializer_336570
є
NoOpNoOp^StatefulPartitionedCall_10^StatefulPartitionedCall_11^StatefulPartitionedCall_12^StatefulPartitionedCall_13^StatefulPartitionedCall_14^StatefulPartitionedCall_15^Variable/Assign^Variable_1/Assign^Variable_2/Assign
Х`
Const_18Const"/device:CPU:0*
_output_shapes
: *
dtype0*§_
valueѓ_B№_ Bщ_
у
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

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses* 

$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses* 

*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
І
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias*
І
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias*
І
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias*
Д
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
А
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

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

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

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

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

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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_6/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_6/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

>0
?1*

>0
?1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_7/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_7/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_8/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_8/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 
y
	_imported
_wrapped_function
 _structured_inputs
Ё_structured_outputs
Ђ_output_to_inputs_map* 
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
Ѓ0
Є1*
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
Ѕ1
І2
Ї3
Ј4
Љ5
Њ6
Ћ7
Ќ8
­9
Ў10
Џ11
А12*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
4
Ѕ0
Ї1
Љ2
Ћ3
­4
Џ5*
4
І0
Ј1
Њ2
Ќ3
Ў4
А5*
V
Бtrace_0
Вtrace_1
Гtrace_2
Дtrace_3
Еtrace_4
Жtrace_5* 
Ћ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
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
Ћ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20* 
Ћ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20* 
Ћ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20* 
Ћ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20* 
Ќ
Щcreated_variables
Ъ	resources
Ыtrackable_objects
Ьinitializers
Эassets
Ю
signatures
$Я_self_saveable_object_factories
transform_fn* 
Ћ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20* 
* 
* 
* 
<
а	variables
б	keras_api

вtotal

гcount*
M
д	variables
е	keras_api

жtotal

зcount
и
_fn_kwargs*
`Z
VARIABLE_VALUEAdam/m/dense_6/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_6/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_6/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_6/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_7/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_7/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/m/dense_7/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/v/dense_7/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_8/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/dense_8/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense_8/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense_8/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
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
й0
к1
л2
м3
н4
о5* 
* 

п0
р1
с2* 

т0
у1
ф2* 

хserving_default* 
* 

в0
г1*

а	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ж0
з1*

д	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
V
п_initializer
ц_create_resource
ч_initialize
ш_destroy_resource* 
V
п_initializer
щ_create_resource
ъ_initialize
ы_destroy_resource* 
V
р_initializer
ь_create_resource
э_initialize
ю_destroy_resource* 
V
р_initializer
я_create_resource
№_initialize
ё_destroy_resource* 
V
с_initializer
ђ_create_resource
ѓ_initialize
є_destroy_resource* 
V
с_initializer
ѕ_create_resource
і_initialize
ї_destroy_resource* 
8
т	_filename
$ј_self_saveable_object_factories* 
8
у	_filename
$љ_self_saveable_object_factories* 
8
ф	_filename
$њ_self_saveable_object_factories* 
* 
* 
* 
Ћ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20* 

ћtrace_0* 

ќtrace_0* 

§trace_0* 

ўtrace_0* 

џtrace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 

trace_0* 
* 
* 
* 
* 

т	capture_0* 
* 
* 

т	capture_0* 
* 
* 

у	capture_0* 
* 
* 

у	capture_0* 
* 
* 

ф	capture_0* 
* 
* 

ф	capture_0* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ж	
StatefulPartitionedCall_16StatefulPartitionedCallsaver_filename"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp)Adam/m/dense_6/kernel/Read/ReadVariableOp)Adam/v/dense_6/kernel/Read/ReadVariableOp'Adam/m/dense_6/bias/Read/ReadVariableOp'Adam/v/dense_6/bias/Read/ReadVariableOp)Adam/m/dense_7/kernel/Read/ReadVariableOp)Adam/v/dense_7/kernel/Read/ReadVariableOp'Adam/m/dense_7/bias/Read/ReadVariableOp'Adam/v/dense_7/bias/Read/ReadVariableOp)Adam/m/dense_8/kernel/Read/ReadVariableOp)Adam/v/dense_8/kernel/Read/ReadVariableOp'Adam/m/dense_8/bias/Read/ReadVariableOp'Adam/v/dense_8/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst_18*%
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_336751
Ю
StatefulPartitionedCall_17StatefulPartitionedCallsaver_filenamedense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias	iterationlearning_rateAdam/m/dense_6/kernelAdam/v/dense_6/kernelAdam/m/dense_6/biasAdam/v/dense_6/biasAdam/m/dense_7/kernelAdam/v/dense_7/kernelAdam/m/dense_7/biasAdam/v/dense_7/biasAdam/m/dense_8/kernelAdam/v/dense_8/kernelAdam/m/dense_8/biasAdam/v/dense_8/biastotal_1count_1totalcount*$
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_336833Њу
ч
r
)__inference_restored_function_body_336446
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
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
GPU2*0J 8 *(
f#R!
__inference__initializer_331506^
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

-
__inference__destroyer_336427
identityћ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336423G
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

-
__inference__destroyer_336542
identityћ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336538G
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

-
__inference__destroyer_331494
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

V
)__inference_restored_function_body_336356
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331490^
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
К(
Р
C__inference_model_1_layer_call_and_return_conditional_losses_336020
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_58
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:H5
'dense_7_biasadd_readvariableop_resource:H8
&dense_8_matmul_readvariableop_resource:H5
'dense_8_biasadd_readvariableop_resource:
identityЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOpЂdense_7/BiasAdd/ReadVariableOpЂdense_7/MatMul/ReadVariableOpЂdense_8/BiasAdd/ReadVariableOpЂdense_8/MatMul/ReadVariableOp[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_4/concatConcatV2inputs_0inputs_1"concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_5/concatConcatV2inputs_2inputs_3inputs_4"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџa
concatenate_6/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :c
concatenate_6/concat/concatIdentityinputs_5*
T0*'
_output_shapes
:џџџџџџџџџ[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :у
concatenate_7/concatConcatV2concatenate_4/concat:output:0concatenate_5/concat:output:0$concatenate_6/concat/concat:output:0"concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_6/MatMulMatMulconcatenate_7/concat:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџH
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџH`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџH
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentitydense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5
Ў
h
.__inference_concatenate_5_layer_call_fn_336108
inputs_0
inputs_1
inputs_2
identityЯ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_335640`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2
З0

T__inference_transform_features_layer_layer_call_and_return_conditional_losses_336350
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

identity_5ЂStatefulPartitionedCallN
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
valueB:б
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
valueB:л
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
:џџџџџџџџџ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџф
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
	2	*
_output_shapes{
y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_pruned_331410k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџm

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:џџџџџџџџџm

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
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
_construction_contextkEagerRuntime*§
_input_shapesы
ш:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:џџџџџџџџџ
3
_user_specified_nameinputs_building_dimension:_[
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameinputs_building_fenced:`\
'
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameinputs_building_painted:]Y
'
_output_shapes
:џџџџџџџџџ
.
_user_specified_nameinputs_building_type:[W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameinputs_customer_id:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameinputs_garden:XT
'
_output_shapes
:џџџџџџџџџ
)
_user_specified_nameinputs_geo_code:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameinputs_insured_period:_[
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameinputs_numberofwindows:Z	V
'
_output_shapes
:џџџџџџџџџ
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
Њ
H
__inference__creator_336436
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336433^
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
І
;
__inference__creator_331546
identityЂ
hash_tableТ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*Э
shared_nameНКhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/26/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_vocabulary', shape=(), dtype=string)_-2_-1_load_331233_331542*
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
ц
Ъ
$__inference_signature_wrapper_335269
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

unknown_20:

unknown_21:

unknown_22:H

unknown_23:H

unknown_24:H

unknown_25:
identityЂStatefulPartitionedCall
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
:џџџџџџџџџ*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *0
f+R)
'__inference_serve_tf_examples_fn_331835o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
#
_output_shapes
:џџџџџџџџџ
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
Л
9
)__inference_restored_function_body_336423
identityа
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
GPU2*0J 8 *&
f!R
__inference__destroyer_331468O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ђ!

$__inference_signature_wrapper_331458

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

identity_6ЂStatefulPartitionedCallи
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
	2	*
_output_shapes{
y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_pruned_331410`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџm

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0	*'
_output_shapes
:џџџџџџџџџm

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*
_input_shapesў
ћ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	inputs_10:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_7:Q	M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_8:Q
M
'
_output_shapes
:џџџџџџџџџ
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
Ј
У
__inference__initializer_331506!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityЂ,text_file_init/InitializeTableFromTextFileV2ѓ
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexўџџџџџџџџ*
value_indexџџџџџџџџџG
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
Т
u
I__inference_concatenate_4_layer_call_and_return_conditional_losses_336101
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
:џџџџџџџџџW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1
Ј
;
__inference__creator_331485
identityЂ
hash_tableФ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*Я
shared_nameПМhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/26/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_2_vocabulary', shape=(), dtype=string)_-2_-1_load_331233_331481*
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
с
V
)__inference_restored_function_body_336627
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331519^
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
У

(__inference_dense_8_layer_call_fn_336191

inputs
unknown:H
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_335705o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџH: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs
Ј
;
__inference__creator_331519
identityЂ
hash_tableФ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*Я
shared_nameПМhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/26/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_1_vocabulary', shape=(), dtype=string)_-2_-1_load_331233_331515*
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
Л
9
)__inference_restored_function_body_336384
identityа
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
GPU2*0J 8 *&
f!R
__inference__destroyer_331494O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 


є
C__inference_dense_6_layer_call_and_return_conditional_losses_336162

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
И
O
#__inference__update_step_xla_336083
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:H: *
	_noinline(:H D

_output_shapes

:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
с
V
)__inference_restored_function_body_336633
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331551^
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
Ј
У
__inference__initializer_331529!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityЂ,text_file_init/InitializeTableFromTextFileV2ѓ
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexўџџџџџџџџ*
value_indexџџџџџџџџџG
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
ч
r
)__inference_restored_function_body_336408
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
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
GPU2*0J 8 *(
f#R!
__inference__initializer_331541^
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

-
__inference__destroyer_336504
identityћ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336500G
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
ч
r
)__inference_restored_function_body_336369
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
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
GPU2*0J 8 *(
f#R!
__inference__initializer_331464^
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
А
L
.__inference_concatenate_6_layer_call_fn_336121
inputs_0
identityЙ
PartitionedCallPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_335648`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Ј
У
__inference__initializer_331464!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityЂ,text_file_init/InitializeTableFromTextFileV2ѓ
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexўџџџџџџџџ*
value_indexџџџџџџџџџG
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

V
)__inference_restored_function_body_336472
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331519^
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
Ќ
K
#__inference__update_step_xla_336088
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
З

I__inference_concatenate_5_layer_call_and_return_conditional_losses_335640

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
:џџџџџџџџџW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Л
9
)__inference_restored_function_body_336461
identityа
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
GPU2*0J 8 *&
f!R
__inference__destroyer_331510O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ч
r
)__inference_restored_function_body_336562
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
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
GPU2*0J 8 *(
f#R!
__inference__initializer_331500^
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
њ!
Ю
C__inference_model_1_layer_call_and_return_conditional_losses_335843

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5 
dense_6_335827:
dense_6_335829: 
dense_7_335832:H
dense_7_335834:H 
dense_8_335837:H
dense_8_335839:
identityЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallа
concatenate_4/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_335630н
concatenate_5/PartitionedCallPartitionedCallinputs_2inputs_3inputs_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_335640Ч
concatenate_6/PartitionedCallPartitionedCallinputs_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_335648З
concatenate_7/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0&concatenate_5/PartitionedCall:output:0&concatenate_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_335658
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_6_335827dense_6_335829*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_335671
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_335832dense_7_335834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_335688
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_335837dense_8_335839*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_335705w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

-
__inference__destroyer_336581
identityћ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336577G
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
Џ
Z
.__inference_concatenate_4_layer_call_fn_336094
inputs_0
inputs_1
identityФ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_335630`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1


є
C__inference_dense_8_layer_call_and_return_conditional_losses_336202

inputs0
matmul_readvariableop_resource:H-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџH: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs

V
)__inference_restored_function_body_336433
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331551^
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
Чf

"__inference__traced_restore_336833
file_prefix1
assignvariableop_dense_6_kernel:-
assignvariableop_1_dense_6_bias:3
!assignvariableop_2_dense_7_kernel:H-
assignvariableop_3_dense_7_bias:H3
!assignvariableop_4_dense_8_kernel:H-
assignvariableop_5_dense_8_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: :
(assignvariableop_8_adam_m_dense_6_kernel::
(assignvariableop_9_adam_v_dense_6_kernel:5
'assignvariableop_10_adam_m_dense_6_bias:5
'assignvariableop_11_adam_v_dense_6_bias:;
)assignvariableop_12_adam_m_dense_7_kernel:H;
)assignvariableop_13_adam_v_dense_7_kernel:H5
'assignvariableop_14_adam_m_dense_7_bias:H5
'assignvariableop_15_adam_v_dense_7_bias:H;
)assignvariableop_16_adam_m_dense_8_kernel:H;
)assignvariableop_17_adam_v_dense_8_kernel:H5
'assignvariableop_18_adam_m_dense_8_bias:5
'assignvariableop_19_adam_v_dense_8_bias:%
assignvariableop_20_total_1: %
assignvariableop_21_count_1: #
assignvariableop_22_total: #
assignvariableop_23_count: 
identity_25ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9§

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѓ

value
B
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЂ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_7_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_7_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_8_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_8_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:Г
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_8AssignVariableOp(assignvariableop_8_adam_m_dense_6_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_9AssignVariableOp(assignvariableop_9_adam_v_dense_6_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_m_dense_6_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_11AssignVariableOp'assignvariableop_11_adam_v_dense_6_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_12AssignVariableOp)assignvariableop_12_adam_m_dense_7_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_v_dense_7_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_m_dense_7_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_v_dense_7_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_m_dense_8_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_v_dense_8_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_m_dense_8_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_19AssignVariableOp'assignvariableop_19_adam_v_dense_8_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:В
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 п
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: Ь
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
Л
9
)__inference_restored_function_body_336500
identityа
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
GPU2*0J 8 *&
f!R
__inference__destroyer_331480O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Н
h
__inference__initializer_336531
unknown
	unknown_0
identityЂStatefulPartitionedCallЃ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336523G
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


є
C__inference_dense_7_layer_call_and_return_conditional_losses_336182

inputs0
matmul_readvariableop_resource:H-
biasadd_readvariableop_resource:H
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџHr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџHP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџHa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ
H
__inference__creator_336552
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336549^
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
У

(__inference_dense_6_layer_call_fn_336151

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_335671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

V
)__inference_restored_function_body_336395
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331546^
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

-
__inference__destroyer_331510
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
ХЗ
Ъ
__inference_pruned_331410

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

identity_6`
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
џџџџџџџџџc
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
valueB: Њ
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:Ј
>scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*:
value1B/ B)x (scale_to_0_1_1/min_and_max/Shape:0) = Њ
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
valueB: Ј
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x == y did not hold element-wise:Є
<scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*8
value/B- B'x (scale_to_0_1/min_and_max/Shape:0) = І
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
 *  ?V
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
valueB"џџџџ   Q
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
 *  ?X
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
valueB"џџџџ   e
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
 *  ?Y
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
џџџџџџџџџg
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
 *  ?[
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
џџџџџџџџџQ
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
 *  ?X
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
valueB"џџџџ   Q
inputs_copyIdentityinputs*
T0*'
_output_shapes
:џџџџџџџџџ
&scale_to_z_score/mean_and_var/IdentityIdentity,scale_to_z_score_mean_and_var_identity_input*
T0*
_output_shapes
: 
scale_to_z_score/subSubinputs_copy:output:0/scale_to_z_score/mean_and_var/Identity:output:0*
T0*'
_output_shapes
:џџџџџџџџџt
scale_to_z_score/zeros_like	ZerosLikescale_to_z_score/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
(scale_to_z_score/mean_and_var/Identity_1Identity.scale_to_z_score_mean_and_var_identity_1_input*
T0*
_output_shapes
: q
scale_to_z_score/SqrtSqrt1scale_to_z_score/mean_and_var/Identity_1:output:0*
T0*
_output_shapes
: 
scale_to_z_score/NotEqualNotEqualscale_to_z_score/Sqrt:y:0$scale_to_z_score/NotEqual/y:output:0*
T0*
_output_shapes
: l
scale_to_z_score/CastCastscale_to_z_score/NotEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
scale_to_z_score/addAddV2scale_to_z_score/zeros_like:y:0scale_to_z_score/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџz
scale_to_z_score/Cast_1Castscale_to_z_score/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:џџџџџџџџџ
scale_to_z_score/truedivRealDivscale_to_z_score/sub:z:0scale_to_z_score/Sqrt:y:0*
T0*'
_output_shapes
:џџџџџџџџџЌ
scale_to_z_score/SelectV2SelectV2scale_to_z_score/Cast_1:y:0scale_to_z_score/truediv:z:0scale_to_z_score/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
	Reshape_2Reshape"scale_to_z_score/SelectV2:output:0Reshape_2/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџU
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:џџџџџџџџџ
Rcompute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2_compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleinputs_1_copy:output:0`compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*&
 _has_manual_control_dependencies(*
_output_shapes
:л
Pcompute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2_compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleS^compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2*&
 _has_manual_control_dependencies(*
_output_shapes
: W
inputs_10_copyIdentity	inputs_10*
T0*'
_output_shapes
:џџџџџџџџџЂ
Tcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2acompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleinputs_10_copy:output:0bcompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*&
 _has_manual_control_dependencies(*
_output_shapes
:с
Rcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2acompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleU^compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2*&
 _has_manual_control_dependencies(*
_output_shapes
: Г
/scale_to_0_1_1/min_and_max/assert_equal_1/EqualEqual)scale_to_0_1_1/min_and_max/Shape:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: Л
-scale_to_0_1_1/min_and_max/assert_equal_1/AllAll3scale_to_0_1_1/min_and_max/assert_equal_1/Equal:z:08scale_to_0_1_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: ­
-scale_to_0_1/min_and_max/assert_equal_1/EqualEqual'scale_to_0_1/min_and_max/Shape:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T0*
_output_shapes
: Е
+scale_to_0_1/min_and_max/assert_equal_1/AllAll1scale_to_0_1/min_and_max/assert_equal_1/Equal:z:06scale_to_0_1/min_and_max/assert_equal_1/Const:output:0*
_output_shapes
: ь
5scale_to_0_1/min_and_max/assert_equal_1/Assert/AssertAssert4scale_to_0_1/min_and_max/assert_equal_1/All:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0'scale_to_0_1/min_and_max/Shape:output:0Escale_to_0_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0)scale_to_0_1/min_and_max/Shape_1:output:0*
T	
2*&
 _has_manual_control_dependencies(*
_output_shapes
 В
7scale_to_0_1_1/min_and_max/assert_equal_1/Assert/AssertAssert6scale_to_0_1_1/min_and_max/assert_equal_1/All:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_0:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_1:output:0)scale_to_0_1_1/min_and_max/Shape:output:0Gscale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert/data_3:output:0+scale_to_0_1_1/min_and_max/Shape_1:output:06^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert*
T	
2*&
 _has_manual_control_dependencies(*
_output_shapes
 U
inputs_2_copyIdentityinputs_2*
T0*'
_output_shapes
:џџџџџџџџџЁ
Tcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2LookupTableFindV2acompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleinputs_2_copy:output:0bcompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*&
 _has_manual_control_dependencies(*
_output_shapes
:с
Rcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2LookupTableSizeV2acompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_table_handleU^compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2*&
 _has_manual_control_dependencies(*
_output_shapes
: р
NoOpNoOpS^compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2Q^compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2U^compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2S^compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2U^compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2S^compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Size/LookupTableSizeV26^scale_to_0_1/min_and_max/assert_equal_1/Assert/Assert8^scale_to_0_1_1/min_and_max/assert_equal_1/Assert/Assert*"
_acd_function_control_output(*&
 _has_manual_control_dependencies(*
_output_shapes
 ]
IdentityIdentityReshape_2:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџЋ
=compute_and_apply_vocabulary/apply_vocab/None_Lookup/NotEqualNotEqual[compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0`compute_and_apply_vocabulary_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*
_output_shapes
:Ў
@compute_and_apply_vocabulary/apply_vocab/None_Lookup/hash_bucketStringToHashBucketFastinputs_1_copy:output:0*'
_output_shapes
:џџџџџџџџџ*
num_buckets
8compute_and_apply_vocabulary/apply_vocab/None_Lookup/AddAddV2Icompute_and_apply_vocabulary/apply_vocab/None_Lookup/hash_bucket:output:0Wcompute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*'
_output_shapes
:џџџџџџџџџЪ
=compute_and_apply_vocabulary/apply_vocab/None_Lookup/SelectV2SelectV2Acompute_and_apply_vocabulary/apply_vocab/None_Lookup/NotEqual:z:0[compute_and_apply_vocabulary/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0<compute_and_apply_vocabulary/apply_vocab/None_Lookup/Add:z:0*
T0	*
_output_shapes
:Ы
one_hotOneHotFcompute_and_apply_vocabulary/apply_vocab/None_Lookup/SelectV2:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
_output_shapes
:r
	Reshape_3Reshapeone_hot:output:0Reshape_3/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_1IdentityReshape_3:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџБ
?compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/NotEqualNotEqual]compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0bcompute_and_apply_vocabulary_1_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*
_output_shapes
:А
Bcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/hash_bucketStringToHashBucketFastinputs_2_copy:output:0*'
_output_shapes
:џџџџџџџџџ*
num_buckets
:compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/AddAddV2Kcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/hash_bucket:output:0Ycompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*'
_output_shapes
:џџџџџџџџџв
?compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/SelectV2SelectV2Ccompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/NotEqual:z:0]compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0>compute_and_apply_vocabulary_1/apply_vocab/None_Lookup/Add:z:0*
T0	*
_output_shapes
:е
	one_hot_1OneHotHcompute_and_apply_vocabulary_1/apply_vocab/None_Lookup/SelectV2:output:0one_hot_1/depth:output:0one_hot_1/on_value:output:0one_hot_1/off_value:output:0*
T0*
_output_shapes
:t
	Reshape_4Reshapeone_hot_1:output:0Reshape_4/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_2IdentityReshape_4:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџU
inputs_3_copyIdentityinputs_3*
T0	*'
_output_shapes
:џџџџџџџџџr
scale_to_0_1/CastCastinputs_3_copy:output:0*

DstT0*

SrcT0	*'
_output_shapes
:џџџџџџџџџ{
#scale_to_0_1/min_and_max/Identity_2Identity)scale_to_0_1_min_and_max_identity_2_input*
T0*
_output_shapes
: 
scale_to_0_1/min_and_max/sub_1Sub)scale_to_0_1/min_and_max/sub_1/x:output:0,scale_to_0_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: 
scale_to_0_1/subSubscale_to_0_1/Cast:y:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџl
scale_to_0_1/zeros_like	ZerosLikescale_to_0_1/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ{
#scale_to_0_1/min_and_max/Identity_3Identity)scale_to_0_1_min_and_max_identity_3_input*
T0*
_output_shapes
: 
scale_to_0_1/LessLess"scale_to_0_1/min_and_max/sub_1:z:0,scale_to_0_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: b
scale_to_0_1/Cast_1Castscale_to_0_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
scale_to_0_1/addAddV2scale_to_0_1/zeros_like:y:0scale_to_0_1/Cast_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџr
scale_to_0_1/Cast_2Castscale_to_0_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:џџџџџџџџџ
scale_to_0_1/sub_1Sub,scale_to_0_1/min_and_max/Identity_3:output:0"scale_to_0_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 
scale_to_0_1/truedivRealDivscale_to_0_1/sub:z:0scale_to_0_1/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџh
scale_to_0_1/SigmoidSigmoidscale_to_0_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ 
scale_to_0_1/SelectV2SelectV2scale_to_0_1/Cast_2:y:0scale_to_0_1/truediv:z:0scale_to_0_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
scale_to_0_1/mulMulscale_to_0_1/SelectV2:output:0scale_to_0_1/mul/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
scale_to_0_1/add_1AddV2scale_to_0_1/mul:z:0scale_to_0_1/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџp
ReshapeReshapescale_to_0_1/add_1:z:0Reshape/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ]

Identity_3IdentityReshape:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџU
inputs_4_copyIdentityinputs_4*
T0	*'
_output_shapes
:џџџџџџџџџg

Identity_4Identityinputs_4_copy:output:0^NoOp*
T0	*'
_output_shapes
:џџџџџџџџџU
inputs_8_copyIdentityinputs_8*
T0*'
_output_shapes
:џџџџџџџџџ
%scale_to_0_1_1/min_and_max/Identity_2Identity+scale_to_0_1_1_min_and_max_identity_2_input*
T0*
_output_shapes
: Ѕ
 scale_to_0_1_1/min_and_max/sub_1Sub+scale_to_0_1_1/min_and_max/sub_1/x:output:0.scale_to_0_1_1/min_and_max/Identity_2:output:0*
T0*
_output_shapes
: 
scale_to_0_1_1/subSubinputs_8_copy:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџp
scale_to_0_1_1/zeros_like	ZerosLikescale_to_0_1_1/sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
%scale_to_0_1_1/min_and_max/Identity_3Identity+scale_to_0_1_1_min_and_max_identity_3_input*
T0*
_output_shapes
: 
scale_to_0_1_1/LessLess$scale_to_0_1_1/min_and_max/sub_1:z:0.scale_to_0_1_1/min_and_max/Identity_3:output:0*
T0*
_output_shapes
: d
scale_to_0_1_1/CastCastscale_to_0_1_1/Less:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
scale_to_0_1_1/addAddV2scale_to_0_1_1/zeros_like:y:0scale_to_0_1_1/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџv
scale_to_0_1_1/Cast_1Castscale_to_0_1_1/add:z:0*

DstT0
*

SrcT0*'
_output_shapes
:џџџџџџџџџ
scale_to_0_1_1/sub_1Sub.scale_to_0_1_1/min_and_max/Identity_3:output:0$scale_to_0_1_1/min_and_max/sub_1:z:0*
T0*
_output_shapes
: 
scale_to_0_1_1/truedivRealDivscale_to_0_1_1/sub:z:0scale_to_0_1_1/sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџk
scale_to_0_1_1/SigmoidSigmoidinputs_8_copy:output:0*
T0*'
_output_shapes
:џџџџџџџџџЈ
scale_to_0_1_1/SelectV2SelectV2scale_to_0_1_1/Cast_1:y:0scale_to_0_1_1/truediv:z:0scale_to_0_1_1/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
scale_to_0_1_1/mulMul scale_to_0_1_1/SelectV2:output:0scale_to_0_1_1/mul/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
scale_to_0_1_1/add_1AddV2scale_to_0_1_1/mul:z:0scale_to_0_1_1/add_1/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџv
	Reshape_1Reshapescale_to_0_1_1/add_1:z:0Reshape_1/shape:output:0*
T0*#
_output_shapes
:џџџџџџџџџ_

Identity_5IdentityReshape_1:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџБ
?compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/NotEqualNotEqual]compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0bcompute_and_apply_vocabulary_2_apply_vocab_none_lookup_none_lookup_lookuptablefindv2_default_value*
T0	*
_output_shapes
:Б
Bcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/hash_bucketStringToHashBucketFastinputs_10_copy:output:0*'
_output_shapes
:џџџџџџџџџ*
num_buckets
:compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/AddAddV2Kcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/hash_bucket:output:0Ycompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Size/LookupTableSizeV2:size:0*
T0	*'
_output_shapes
:џџџџџџџџџв
?compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/SelectV2SelectV2Ccompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/NotEqual:z:0]compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/None_Lookup/LookupTableFindV2:values:0>compute_and_apply_vocabulary_2/apply_vocab/None_Lookup/Add:z:0*
T0	*
_output_shapes
:е
	one_hot_2OneHotHcompute_and_apply_vocabulary_2/apply_vocab/None_Lookup/SelectV2:output:0one_hot_2/depth:output:0one_hot_2/on_value:output:0one_hot_2/off_value:output:0*
T0*
_output_shapes
:t
	Reshape_5Reshapeone_hot_2:output:0Reshape_5/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџc

Identity_6IdentityReshape_5:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0*(
_construction_contextkEagerRuntime*
_input_shapesў
ћ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : :- )
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-)
'
_output_shapes
:џџџџџџџџџ:-	)
'
_output_shapes
:џџџџџџџџџ:-
)
'
_output_shapes
:џџџџџџџџџ:
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
#
љ
9__inference_transform_features_layer_layer_call_fn_336268
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

identity_5ЂStatefulPartitionedCall
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
f:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_335403k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџm

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:џџџџџџџџџm

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
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
_construction_contextkEagerRuntime*§
_input_shapesы
ш:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:b ^
'
_output_shapes
:џџџџџџџџџ
3
_user_specified_nameinputs_building_dimension:_[
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameinputs_building_fenced:`\
'
_output_shapes
:џџџџџџџџџ
1
_user_specified_nameinputs_building_painted:]Y
'
_output_shapes
:џџџџџџџџџ
.
_user_specified_nameinputs_building_type:[W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameinputs_customer_id:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameinputs_garden:XT
'
_output_shapes
:џџџџџџџџџ
)
_user_specified_nameinputs_geo_code:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameinputs_insured_period:_[
'
_output_shapes
:џџџџџџџџџ
0
_user_specified_nameinputs_numberofwindows:Z	V
'
_output_shapes
:џџџџџџџџџ
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
І
;
__inference__creator_331490
identityЂ
hash_tableТ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*Э
shared_nameНКhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/26/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_vocabulary', shape=(), dtype=string)_-2_-1_load_331233_331486*
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
Т

I__inference_concatenate_5_layer_call_and_return_conditional_losses_336116
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2
К
s
I__inference_concatenate_4_layer_call_and_return_conditional_losses_335630

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
:џџџџџџџџџW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П.
Щ
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_335606
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

identity_5ЂStatefulPartitionedCall@
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
valueB:б
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
valueB:л
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
:џџџџџџџџџ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџ
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
	2	*
_output_shapes{
y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_pruned_331410k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџm

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:џџџџџџџџџm

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
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
_construction_contextkEagerRuntime*§
_input_shapesы
ш:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameBuilding Dimension:XT
'
_output_shapes
:џџџџџџџџџ
)
_user_specified_nameBuilding_Fenced:YU
'
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameBuilding_Painted:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameBuilding_Type:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameCustomer Id:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameGarden:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
Geo_Code:WS
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameInsured_Period:XT
'
_output_shapes
:џџџџџџџџџ
)
_user_specified_nameNumberOfWindows:S	O
'
_output_shapes
:џџџџџџџџџ
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


є
C__inference_dense_7_layer_call_and_return_conditional_losses_335688

inputs0
matmul_readvariableop_resource:H-
biasadd_readvariableop_resource:H
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџHr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџHP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџHa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџHw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Д.
Ќ
!__inference__wrapped_model_335308
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder@
.model_1_dense_6_matmul_readvariableop_resource:=
/model_1_dense_6_biasadd_readvariableop_resource:@
.model_1_dense_7_matmul_readvariableop_resource:H=
/model_1_dense_7_biasadd_readvariableop_resource:H@
.model_1_dense_8_matmul_readvariableop_resource:H=
/model_1_dense_8_biasadd_readvariableop_resource:
identityЂ&model_1/dense_6/BiasAdd/ReadVariableOpЂ%model_1/dense_6/MatMul/ReadVariableOpЂ&model_1/dense_7/BiasAdd/ReadVariableOpЂ%model_1/dense_7/MatMul/ReadVariableOpЂ&model_1/dense_8/BiasAdd/ReadVariableOpЂ%model_1/dense_8/MatMul/ReadVariableOpc
!model_1/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
model_1/concatenate_4/concatConcatV2building_type_xfinsured_period_xf*model_1/concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџc
!model_1/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ч
model_1/concatenate_5/concatConcatV2building_fenced_xfbuilding_painted_xfsettlement_xf*model_1/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџi
'model_1/concatenate_6/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :n
#model_1/concatenate_6/concat/concatIdentityplaceholder*
T0*'
_output_shapes
:џџџџџџџџџc
!model_1/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
model_1/concatenate_7/concatConcatV2%model_1/concatenate_4/concat:output:0%model_1/concatenate_5/concat:output:0,model_1/concatenate_6/concat/concat:output:0*model_1/concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ј
model_1/dense_6/MatMulMatMul%model_1/concatenate_7/concat:output:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
model_1/dense_6/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0Ѕ
model_1/dense_7/MatMulMatMul"model_1/dense_6/Relu:activations:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџH
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0І
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџHp
model_1/dense_7/ReluRelu model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџH
%model_1/dense_8/MatMul/ReadVariableOpReadVariableOp.model_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0Ѕ
model_1/dense_8/MatMulMatMul"model_1/dense_7/Relu:activations:0-model_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
model_1/dense_8/BiasAddBiasAdd model_1/dense_8/MatMul:product:0.model_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
model_1/dense_8/SigmoidSigmoid model_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentitymodel_1/dense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЙ
NoOpNoOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp'^model_1/dense_8/BiasAdd/ReadVariableOp&^model_1/dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp2P
&model_1/dense_8/BiasAdd/ReadVariableOp&model_1/dense_8/BiasAdd/ReadVariableOp2N
%model_1/dense_8/MatMul/ReadVariableOp%model_1/dense_8/MatMul/ReadVariableOp:Y U
'
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameBuilding Dimension_xf

-
__inference__destroyer_336465
identityћ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336461G
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

-
__inference__destroyer_331480
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
ч
r
)__inference_restored_function_body_336523
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
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
GPU2*0J 8 *(
f#R!
__inference__initializer_331529^
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
Н
h
__inference__initializer_336416
unknown
	unknown_0
identityЂStatefulPartitionedCallЃ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336408G
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
Њ
H
__inference__creator_336513
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336510^
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

-
__inference__destroyer_331468
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
Њ
H
__inference__creator_336398
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336395^
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

ї
(__inference_model_1_layer_call_fn_335727
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder
unknown:
	unknown_0:
	unknown_1:H
	unknown_2:H
	unknown_3:H
	unknown_4:
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallbuilding_type_xfinsured_period_xfbuilding_fenced_xfbuilding_painted_xfsettlement_xfplaceholderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_335712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameBuilding Dimension_xf
Ќ
K
#__inference__update_step_xla_336068
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

V
)__inference_restored_function_body_336510
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331255^
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
-

T__inference_transform_features_layer_layer_call_and_return_conditional_losses_335403

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

identity_5ЂStatefulPartitionedCall;
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
valueB:б
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
valueB:л
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
:џџџџџџџџџ
PlaceholderWithDefaultPlaceholderWithDefaultzeros:output:0*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџю
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
	2	*
_output_shapes{
y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_pruned_331410k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџm

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:џџџџџџџџџm

Identity_4Identity StatefulPartitionedCall:output:5^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
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
_construction_contextkEagerRuntime*§
_input_shapesы
ш:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:O	K
'
_output_shapes
:џџџџџџџџџ
 
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
У

(__inference_dense_7_layer_call_fn_336171

inputs
unknown:H
	unknown_0:H
identityЂStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_335688o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџH`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
У
__inference__initializer_331535!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityЂ,text_file_init/InitializeTableFromTextFileV2ѓ
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexўџџџџџџџџ*
value_indexџџџџџџџџџG
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
Ј
У
__inference__initializer_331500!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityЂ,text_file_init/InitializeTableFromTextFileV2ѓ
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexўџџџџџџџџ*
value_indexџџџџџџџџџG
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
Ј
;
__inference__creator_331255
identityЂ
hash_tableФ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*Я
shared_nameПМhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/26/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_2_vocabulary', shape=(), dtype=string)_-2_-1_load_331233_331251*
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
Л
9
)__inference_restored_function_body_336538
identityа
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
GPU2*0J 8 *&
f!R
__inference__destroyer_331514O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ю
Щ
(__inference_model_1_layer_call_fn_335982
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown:
	unknown_0:
	unknown_1:H
	unknown_2:H
	unknown_3:H
	unknown_4:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_335843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5
Н
h
__inference__initializer_336377
unknown
	unknown_0
identityЂStatefulPartitionedCallЃ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336369G
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
Ј
У
__inference__initializer_331541!
text_file_init_asset_filepath=
9text_file_init_initializetablefromtextfilev2_table_handle
identityЂ,text_file_init/InitializeTableFromTextFileV2ѓ
,text_file_init/InitializeTableFromTextFileV2InitializeTableFromTextFileV29text_file_init_initializetablefromtextfilev2_table_handletext_file_init_asset_filepath*
_output_shapes
 *
	key_indexўџџџџџџџџ*
value_indexџџџџџџџџџG
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
Њ!
Ў
9__inference_transform_features_layer_layer_call_fn_335458
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

identity_5ЂStatefulPartitionedCallМ
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
f:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_335403k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџm

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*#
_output_shapes
:џџџџџџџџџm

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
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
_construction_contextkEagerRuntime*§
_input_shapesы
ш:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameBuilding Dimension:XT
'
_output_shapes
:џџџџџџџџџ
)
_user_specified_nameBuilding_Fenced:YU
'
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameBuilding_Painted:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameBuilding_Type:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameCustomer Id:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameGarden:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
Geo_Code:WS
'
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameInsured_Period:XT
'
_output_shapes
:џџџџџџџџџ
)
_user_specified_nameNumberOfWindows:S	O
'
_output_shapes
:џџџџџџџџџ
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
Ј
;
__inference__creator_331551
identityЂ
hash_tableФ

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*Я
shared_nameПМhash_table_tf.Tensor(b'./pipeline/Transform/transform_graph/26/.temp_path/tftransform_tmp/vocab_compute_and_apply_vocabulary_1_vocabulary', shape=(), dtype=string)_-2_-1_load_331233_331547*
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

-
__inference__destroyer_331523
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
њ!
Ю
C__inference_model_1_layer_call_and_return_conditional_losses_335712

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5 
dense_6_335672:
dense_6_335674: 
dense_7_335689:H
dense_7_335691:H 
dense_8_335706:H
dense_8_335708:
identityЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallа
concatenate_4/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_335630н
concatenate_5/PartitionedCallPartitionedCallinputs_2inputs_3inputs_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_335640Ч
concatenate_6/PartitionedCallPartitionedCallinputs_5*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_335648З
concatenate_7/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0&concatenate_5/PartitionedCall:output:0&concatenate_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_335658
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_6_335672dense_6_335674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_335671
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_335689dense_7_335691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_335688
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_335706dense_8_335708*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_335705w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ
H
__inference__creator_336475
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336472^
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
ю
Щ
(__inference_model_1_layer_call_fn_335960
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
unknown:
	unknown_0:
	unknown_1:H
	unknown_2:H
	unknown_3:H
	unknown_4:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_335712o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5

-
__inference__destroyer_331514
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
ч
r
)__inference_restored_function_body_336485
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
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
GPU2*0J 8 *(
f#R!
__inference__initializer_331535^
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
Њ
H
__inference__creator_336359
identityЂStatefulPartitionedCall
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336356^
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
И
O
#__inference__update_step_xla_336063
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:: *
	_noinline(:H D

_output_shapes

:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ќ
K
#__inference__update_step_xla_336078
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:H: *
	_noinline(:D @

_output_shapes
:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Й
g
I__inference_concatenate_6_layer_call_and_return_conditional_losses_336127
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
:џџџџџџџџџ^
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Г
e
I__inference_concatenate_6_layer_call_and_return_conditional_losses_335648

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
:џџџџџџџџџ^
IdentityIdentityconcat/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
с
V
)__inference_restored_function_body_336622
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331255^
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
Є5
ѓ	
__inference__traced_save_336751
file_prefix-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop4
0savev2_adam_m_dense_6_kernel_read_readvariableop4
0savev2_adam_v_dense_6_kernel_read_readvariableop2
.savev2_adam_m_dense_6_bias_read_readvariableop2
.savev2_adam_v_dense_6_bias_read_readvariableop4
0savev2_adam_m_dense_7_kernel_read_readvariableop4
0savev2_adam_v_dense_7_kernel_read_readvariableop2
.savev2_adam_m_dense_7_bias_read_readvariableop2
.savev2_adam_v_dense_7_bias_read_readvariableop4
0savev2_adam_m_dense_8_kernel_read_readvariableop4
0savev2_adam_v_dense_8_kernel_read_readvariableop2
.savev2_adam_m_dense_8_bias_read_readvariableop2
.savev2_adam_v_dense_8_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const_18

identity_1ЂMergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: њ

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ѓ

value
B
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop0savev2_adam_m_dense_6_kernel_read_readvariableop0savev2_adam_v_dense_6_kernel_read_readvariableop.savev2_adam_m_dense_6_bias_read_readvariableop.savev2_adam_v_dense_6_bias_read_readvariableop0savev2_adam_m_dense_7_kernel_read_readvariableop0savev2_adam_v_dense_7_kernel_read_readvariableop.savev2_adam_m_dense_7_bias_read_readvariableop.savev2_adam_v_dense_7_bias_read_readvariableop0savev2_adam_m_dense_8_kernel_read_readvariableop0savev2_adam_v_dense_8_kernel_read_readvariableop.savev2_adam_m_dense_8_bias_read_readvariableop.savev2_adam_v_dense_8_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const_18"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *'
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
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

identity_1Identity_1:output:0*Е
_input_shapesЃ
 : :::H:H:H:: : :::::H:H:H:H:H:H::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:H: 

_output_shapes
:H:$ 

_output_shapes

:H: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

::$
 

_output_shapes

:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:H:$ 

_output_shapes

:H: 

_output_shapes
:H: 

_output_shapes
:H:$ 

_output_shapes

:H:$ 

_output_shapes

:H: 
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
Н
h
__inference__initializer_336454
unknown
	unknown_0
identityЂStatefulPartitionedCallЃ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336446G
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


є
C__inference_dense_6_layer_call_and_return_conditional_losses_335671

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў
h
.__inference_concatenate_7_layer_call_fn_336134
inputs_0
inputs_1
inputs_2
identityЯ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_335658`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2

-
__inference__destroyer_336388
identityћ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336384G
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
с
V
)__inference_restored_function_body_336644
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331490^
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
И
O
#__inference__update_step_xla_336073
gradient
variable:H*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:H: *
	_noinline(:H D

_output_shapes

:H
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
с
V
)__inference_restored_function_body_336616
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331485^
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
щo
њ
'__inference_serve_tf_examples_fn_331835
examples#
transform_features_layer_331750#
transform_features_layer_331752#
transform_features_layer_331754#
transform_features_layer_331756#
transform_features_layer_331758#
transform_features_layer_331760#
transform_features_layer_331762	#
transform_features_layer_331764	#
transform_features_layer_331766#
transform_features_layer_331768	#
transform_features_layer_331770	#
transform_features_layer_331772	#
transform_features_layer_331774	#
transform_features_layer_331776#
transform_features_layer_331778	#
transform_features_layer_331780	#
transform_features_layer_331782	#
transform_features_layer_331784	#
transform_features_layer_331786#
transform_features_layer_331788	#
transform_features_layer_331790	@
.model_1_dense_6_matmul_readvariableop_resource:=
/model_1_dense_6_biasadd_readvariableop_resource:@
.model_1_dense_7_matmul_readvariableop_resource:H=
/model_1_dense_7_biasadd_readvariableop_resource:H@
.model_1_dense_8_matmul_readvariableop_resource:H=
/model_1_dense_8_biasadd_readvariableop_resource:
identityЂ&model_1/dense_6/BiasAdd/ReadVariableOpЂ%model_1/dense_6/MatMul/ReadVariableOpЂ&model_1/dense_7/BiasAdd/ReadVariableOpЂ%model_1/dense_7/MatMul/ReadVariableOpЂ&model_1/dense_8/BiasAdd/ReadVariableOpЂ%model_1/dense_8/MatMul/ReadVariableOpЂ0transform_features_layer/StatefulPartitionedCallU
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
valueB 
&ParseExample/ParseExampleV2/dense_keysConst*
_output_shapes
:
*
dtype0*Ї
valueB
BBuilding DimensionBBuilding_FencedBBuilding_PaintedBBuilding_TypeBCustomer IdBGardenBGeo_CodeBInsured_PeriodBNumberOfWindowsB
Settlementj
'ParseExample/ParseExampleV2/ragged_keysConst*
_output_shapes
: *
dtype0*
valueB Ч
ParseExample/ParseExampleV2ParseExampleV2examples*ParseExample/ParseExampleV2/names:output:00ParseExample/ParseExampleV2/sparse_keys:output:0/ParseExample/ParseExampleV2/dense_keys:output:00ParseExample/ParseExampleV2/ragged_keys:output:0ParseExample/Const:output:0ParseExample/Const_1:output:0ParseExample/Const_2:output:0ParseExample/Const_3:output:0ParseExample/Const_4:output:0ParseExample/Const_5:output:0ParseExample/Const_6:output:0ParseExample/Const_7:output:0ParseExample/Const_8:output:0ParseExample/Const_9:output:0*
Tdense
2
	*д
_output_shapesС
О:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*N
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
valueB:Ю
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
valueB:и
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
value	B :Р
%transform_features_layer/zeros/packedPack1transform_features_layer/strided_slice_1:output:00transform_features_layer/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
$transform_features_layer/zeros/ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R З
transform_features_layer/zerosFill.transform_features_layer/zeros/packed:output:0-transform_features_layer/zeros/Const:output:0*
T0	*'
_output_shapes
:џџџџџџџџџЦ
/transform_features_layer/PlaceholderWithDefaultPlaceholderWithDefault'transform_features_layer/zeros:output:0*'
_output_shapes
:џџџџџџџџџ*
dtype0	*
shape:џџџџџџџџџМ
0transform_features_layer/StatefulPartitionedCallStatefulPartitionedCall*ParseExample/ParseExampleV2:dense_values:0*ParseExample/ParseExampleV2:dense_values:1*ParseExample/ParseExampleV2:dense_values:2*ParseExample/ParseExampleV2:dense_values:38transform_features_layer/PlaceholderWithDefault:output:0*ParseExample/ParseExampleV2:dense_values:4*ParseExample/ParseExampleV2:dense_values:5*ParseExample/ParseExampleV2:dense_values:6*ParseExample/ParseExampleV2:dense_values:7*ParseExample/ParseExampleV2:dense_values:8*ParseExample/ParseExampleV2:dense_values:9transform_features_layer_331750transform_features_layer_331752transform_features_layer_331754transform_features_layer_331756transform_features_layer_331758transform_features_layer_331760transform_features_layer_331762transform_features_layer_331764transform_features_layer_331766transform_features_layer_331768transform_features_layer_331770transform_features_layer_331772transform_features_layer_331774transform_features_layer_331776transform_features_layer_331778transform_features_layer_331780transform_features_layer_331782transform_features_layer_331784transform_features_layer_331786transform_features_layer_331788transform_features_layer_331790*+
Tin$
"2 														*
Tout
	2	*
_output_shapes{
y:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_pruned_331410a
model_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџЎ
model_1/ExpandDims
ExpandDims9transform_features_layer/StatefulPartitionedCall:output:3model_1/ExpandDims/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
model_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
model_1/ExpandDims_1
ExpandDims9transform_features_layer/StatefulPartitionedCall:output:5!model_1/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
model_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџВ
model_1/ExpandDims_2
ExpandDims9transform_features_layer/StatefulPartitionedCall:output:0!model_1/ExpandDims_2/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!model_1/concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ы
model_1/concatenate_4/concatConcatV2model_1/ExpandDims:output:0model_1/ExpandDims_1:output:0*model_1/concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџc
!model_1/concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Р
model_1/concatenate_5/concatConcatV29transform_features_layer/StatefulPartitionedCall:output:19transform_features_layer/StatefulPartitionedCall:output:29transform_features_layer/StatefulPartitionedCall:output:6*model_1/concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџi
'model_1/concatenate_6/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :
#model_1/concatenate_6/concat/concatIdentitymodel_1/ExpandDims_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџc
!model_1/concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
model_1/concatenate_7/concatConcatV2%model_1/concatenate_4/concat:output:0%model_1/concatenate_5/concat:output:0,model_1/concatenate_6/concat/concat:output:0*model_1/concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
%model_1/dense_6/MatMul/ReadVariableOpReadVariableOp.model_1_dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ј
model_1/dense_6/MatMulMatMul%model_1/concatenate_7/concat:output:0-model_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
model_1/dense_6/BiasAddBiasAdd model_1/dense_6/MatMul:product:0.model_1/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
model_1/dense_6/ReluRelu model_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
%model_1/dense_7/MatMul/ReadVariableOpReadVariableOp.model_1_dense_7_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0Ѕ
model_1/dense_7/MatMulMatMul"model_1/dense_6/Relu:activations:0-model_1/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџH
&model_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_7_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0І
model_1/dense_7/BiasAddBiasAdd model_1/dense_7/MatMul:product:0.model_1/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџHp
model_1/dense_7/ReluRelu model_1/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџH
%model_1/dense_8/MatMul/ReadVariableOpReadVariableOp.model_1_dense_8_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0Ѕ
model_1/dense_8/MatMulMatMul"model_1/dense_7/Relu:activations:0-model_1/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model_1/dense_8/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0І
model_1/dense_8/BiasAddBiasAdd model_1/dense_8/MatMul:product:0.model_1/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџv
model_1/dense_8/SigmoidSigmoid model_1/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentitymodel_1/dense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџь
NoOpNoOp'^model_1/dense_6/BiasAdd/ReadVariableOp&^model_1/dense_6/MatMul/ReadVariableOp'^model_1/dense_7/BiasAdd/ReadVariableOp&^model_1/dense_7/MatMul/ReadVariableOp'^model_1/dense_8/BiasAdd/ReadVariableOp&^model_1/dense_8/MatMul/ReadVariableOp1^transform_features_layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*X
_input_shapesG
E:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&model_1/dense_6/BiasAdd/ReadVariableOp&model_1/dense_6/BiasAdd/ReadVariableOp2N
%model_1/dense_6/MatMul/ReadVariableOp%model_1/dense_6/MatMul/ReadVariableOp2P
&model_1/dense_7/BiasAdd/ReadVariableOp&model_1/dense_7/BiasAdd/ReadVariableOp2N
%model_1/dense_7/MatMul/ReadVariableOp%model_1/dense_7/MatMul/ReadVariableOp2P
&model_1/dense_8/BiasAdd/ReadVariableOp&model_1/dense_8/BiasAdd/ReadVariableOp2N
%model_1/dense_8/MatMul/ReadVariableOp%model_1/dense_8/MatMul/ReadVariableOp2d
0transform_features_layer/StatefulPartitionedCall0transform_features_layer/StatefulPartitionedCall:M I
#
_output_shapes
:џџџџџџџџџ
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
З

I__inference_concatenate_7_layer_call_and_return_conditional_losses_335658

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
:џџџџџџџџџW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

ї
(__inference_model_1_layer_call_fn_335880
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder
unknown:
	unknown_0:
	unknown_1:H
	unknown_2:H
	unknown_3:H
	unknown_4:
identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallbuilding_type_xfinsured_period_xfbuilding_fenced_xfbuilding_painted_xfsettlement_xfplaceholderunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*(
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_335843o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameBuilding Dimension_xf
Т

I__inference_concatenate_7_layer_call_and_return_conditional_losses_336142
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2
#
ў
C__inference_model_1_layer_call_and_return_conditional_losses_335936
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder 
dense_6_335920:
dense_6_335922: 
dense_7_335925:H
dense_7_335927:H 
dense_8_335930:H
dense_8_335932:
identityЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallу
concatenate_4/PartitionedCallPartitionedCallbuilding_type_xfinsured_period_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_335630ї
concatenate_5/PartitionedCallPartitionedCallbuilding_fenced_xfbuilding_painted_xfsettlement_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_335640Ъ
concatenate_6/PartitionedCallPartitionedCallplaceholder*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_335648З
concatenate_7/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0&concatenate_5/PartitionedCall:output:0&concatenate_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_335658
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_6_335920dense_6_335922*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_335671
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_335925dense_7_335927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_335688
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_335930dense_8_335932*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_335705w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Y U
'
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameBuilding Dimension_xf
Н
h
__inference__initializer_336570
unknown
	unknown_0
identityЂStatefulPartitionedCallЃ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336562G
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
#
ў
C__inference_model_1_layer_call_and_return_conditional_losses_335908
building_type_xf
insured_period_xf
building_fenced_xf
building_painted_xf
settlement_xf
placeholder 
dense_6_335892:
dense_6_335894: 
dense_7_335897:H
dense_7_335899:H 
dense_8_335902:H
dense_8_335904:
identityЂdense_6/StatefulPartitionedCallЂdense_7/StatefulPartitionedCallЂdense_8/StatefulPartitionedCallу
concatenate_4/PartitionedCallPartitionedCallbuilding_type_xfinsured_period_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_4_layer_call_and_return_conditional_losses_335630ї
concatenate_5/PartitionedCallPartitionedCallbuilding_fenced_xfbuilding_painted_xfsettlement_xf*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_5_layer_call_and_return_conditional_losses_335640Ъ
concatenate_6/PartitionedCallPartitionedCallplaceholder*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_6_layer_call_and_return_conditional_losses_335648З
concatenate_7/PartitionedCallPartitionedCall&concatenate_4/PartitionedCall:output:0&concatenate_5/PartitionedCall:output:0&concatenate_6/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_concatenate_7_layer_call_and_return_conditional_losses_335658
dense_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_7/PartitionedCall:output:0dense_6_335892dense_6_335894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_335671
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_335897dense_7_335899*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџH*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_335688
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_335902dense_8_335904*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_8_layer_call_and_return_conditional_losses_335705w
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЌ
NoOpNoOp ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:Y U
'
_output_shapes
:џџџџџџџџџ
*
_user_specified_nameBuilding_Type_xf:ZV
'
_output_shapes
:џџџџџџџџџ
+
_user_specified_nameInsured_Period_xf:[W
'
_output_shapes
:џџџџџџџџџ
,
_user_specified_nameBuilding_Fenced_xf:\X
'
_output_shapes
:џџџџџџџџџ
-
_user_specified_nameBuilding_Painted_xf:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_nameSettlement_xf:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nameBuilding Dimension_xf
с
V
)__inference_restored_function_body_336638
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331546^
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
Л
9
)__inference_restored_function_body_336577
identityа
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
GPU2*0J 8 *&
f!R
__inference__destroyer_331523O
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Н
h
__inference__initializer_336493
unknown
	unknown_0
identityЂStatefulPartitionedCallЃ
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
GPU2*0J 8 *2
f-R+
)__inference_restored_function_body_336485G
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
К(
Р
C__inference_model_1_layer_call_and_return_conditional_losses_336058
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_58
&dense_6_matmul_readvariableop_resource:5
'dense_6_biasadd_readvariableop_resource:8
&dense_7_matmul_readvariableop_resource:H5
'dense_7_biasadd_readvariableop_resource:H8
&dense_8_matmul_readvariableop_resource:H5
'dense_8_biasadd_readvariableop_resource:
identityЂdense_6/BiasAdd/ReadVariableOpЂdense_6/MatMul/ReadVariableOpЂdense_7/BiasAdd/ReadVariableOpЂdense_7/MatMul/ReadVariableOpЂdense_8/BiasAdd/ReadVariableOpЂdense_8/MatMul/ReadVariableOp[
concatenate_4/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_4/concatConcatV2inputs_0inputs_1"concatenate_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ[
concatenate_5/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate_5/concatConcatV2inputs_2inputs_3inputs_4"concatenate_5/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџa
concatenate_6/concat/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :c
concatenate_6/concat/concatIdentityinputs_5*
T0*'
_output_shapes
:џџџџџџџџџ[
concatenate_7/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :у
concatenate_7/concatConcatV2concatenate_4/concat:output:0concatenate_5/concat:output:0$concatenate_6/concat/concat:output:0"concatenate_7/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_6/MatMulMatMulconcatenate_7/concat:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџH
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype0
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџH`
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџH
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes

:H*
dtype0
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџf
dense_8/SigmoidSigmoiddense_8/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџb
IdentityIdentitydense_8/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
~:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : 2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5

V
)__inference_restored_function_body_336549
identityЂStatefulPartitionedCallо
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
GPU2*0J 8 *$
fR
__inference__creator_331485^
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


є
C__inference_dense_8_layer_call_and_return_conditional_losses_335705

inputs0
matmul_readvariableop_resource:H-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:H*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџH: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџH
 
_user_specified_nameinputs"
N
saver_filename:0StatefulPartitionedCall_16:0StatefulPartitionedCall_178"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ћ
serving_default
9
examples-
serving_default_examples:0џџџџџџџџџ>
output_02
StatefulPartitionedCall_9:0џџџџџџџџџtensorflow/serving/predict2M

asset_path_initializer:0/vocab_compute_and_apply_vocabulary_2_vocabulary2O

asset_path_initializer_1:0/vocab_compute_and_apply_vocabulary_1_vocabulary2M

asset_path_initializer_2:0-vocab_compute_and_apply_vocabulary_vocabulary:§­
њ
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
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses

6kernel
7bias"
_tf_keras_layer
Л
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias"
_tf_keras_layer
Л
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses

Fkernel
Gbias"
_tf_keras_layer
Ы
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
Ъ
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
е
Ttrace_0
Utrace_1
Vtrace_2
Wtrace_32ъ
(__inference_model_1_layer_call_fn_335727
(__inference_model_1_layer_call_fn_335960
(__inference_model_1_layer_call_fn_335982
(__inference_model_1_layer_call_fn_335880П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zTtrace_0zUtrace_1zVtrace_2zWtrace_3
С
Xtrace_0
Ytrace_1
Ztrace_2
[trace_32ж
C__inference_model_1_layer_call_and_return_conditional_losses_336020
C__inference_model_1_layer_call_and_return_conditional_losses_336058
C__inference_model_1_layer_call_and_return_conditional_losses_335908
C__inference_model_1_layer_call_and_return_conditional_losses_335936П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zXtrace_0zYtrace_1zZtrace_2z[trace_3
ЗBД
!__inference__wrapped_model_335308Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

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
­
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
ђ
itrace_02е
.__inference_concatenate_4_layer_call_fn_336094Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zitrace_0

jtrace_02№
I__inference_concatenate_4_layer_call_and_return_conditional_losses_336101Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zjtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
ђ
ptrace_02е
.__inference_concatenate_5_layer_call_fn_336108Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zptrace_0

qtrace_02№
I__inference_concatenate_5_layer_call_and_return_conditional_losses_336116Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zqtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
ђ
wtrace_02е
.__inference_concatenate_6_layer_call_fn_336121Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zwtrace_0

xtrace_02№
I__inference_concatenate_6_layer_call_and_return_conditional_losses_336127Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zxtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
ђ
~trace_02е
.__inference_concatenate_7_layer_call_fn_336134Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 z~trace_0

trace_02№
I__inference_concatenate_7_layer_call_and_return_conditional_losses_336142Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_dense_6_layer_call_fn_336151Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_dense_6_layer_call_and_return_conditional_losses_336162Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 :2dense_6/kernel
:2dense_6/bias
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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_dense_7_layer_call_fn_336171Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_dense_7_layer_call_and_return_conditional_losses_336182Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 :H2dense_7/kernel
:H2dense_7/bias
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
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
ю
trace_02Я
(__inference_dense_8_layer_call_fn_336191Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ъ
C__inference_dense_8_layer_call_and_return_conditional_losses_336202Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
 :H2dense_8/kernel
:2dense_8/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
ж
trace_0
trace_12
9__inference_transform_features_layer_layer_call_fn_335458
9__inference_transform_features_layer_layer_call_fn_336268Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

trace_0
trace_12б
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_336350
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_335606Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1

	_imported
_wrapped_function
 _structured_inputs
Ё_structured_outputs
Ђ_output_to_inputs_map"
trackable_dict_wrapper
 "
trackable_list_wrapper

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
Ѓ0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
(__inference_model_1_layer_call_fn_335727Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
­BЊ
(__inference_model_1_layer_call_fn_335960inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
­BЊ
(__inference_model_1_layer_call_fn_335982inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
хBт
(__inference_model_1_layer_call_fn_335880Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ШBХ
C__inference_model_1_layer_call_and_return_conditional_losses_336020inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ШBХ
C__inference_model_1_layer_call_and_return_conditional_losses_336058inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
C__inference_model_1_layer_call_and_return_conditional_losses_335908Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
C__inference_model_1_layer_call_and_return_conditional_losses_335936Building_Type_xfInsured_Period_xfBuilding_Fenced_xfBuilding_Painted_xfSettlement_xfBuilding Dimension_xf"П
ЖВВ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

]0
Ѕ1
І2
Ї3
Ј4
Љ5
Њ6
Ћ7
Ќ8
­9
Ў10
Џ11
А12"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
P
Ѕ0
Ї1
Љ2
Ћ3
­4
Џ5"
trackable_list_wrapper
P
І0
Ј1
Њ2
Ќ3
Ў4
А5"
trackable_list_wrapper
Х
Бtrace_0
Вtrace_1
Гtrace_2
Дtrace_3
Еtrace_4
Жtrace_52
#__inference__update_step_xla_336063
#__inference__update_step_xla_336068
#__inference__update_step_xla_336073
#__inference__update_step_xla_336078
#__inference__update_step_xla_336083
#__inference__update_step_xla_336088Й
ЎВЊ
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0zБtrace_0zВtrace_1zГtrace_2zДtrace_3zЕtrace_4zЖtrace_5

З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20BЩ
$__inference_signature_wrapper_335269examples"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗ	capture_0zИ	capture_1zЙ	capture_2zК	capture_3zЛ	capture_4zМ	capture_5zН	capture_6zО	capture_7zП	capture_9zР
capture_10zС
capture_11zТ
capture_12zУ
capture_14zФ
capture_15zХ
capture_16zЦ
capture_17zЧ
capture_19zШ
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
юBы
.__inference_concatenate_4_layer_call_fn_336094inputs_0inputs_1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_concatenate_4_layer_call_and_return_conditional_losses_336101inputs_0inputs_1"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
јBѕ
.__inference_concatenate_5_layer_call_fn_336108inputs_0inputs_1inputs_2"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_concatenate_5_layer_call_and_return_conditional_losses_336116inputs_0inputs_1inputs_2"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
фBс
.__inference_concatenate_6_layer_call_fn_336121inputs_0"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
I__inference_concatenate_6_layer_call_and_return_conditional_losses_336127inputs_0"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
јBѕ
.__inference_concatenate_7_layer_call_fn_336134inputs_0inputs_1inputs_2"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
I__inference_concatenate_7_layer_call_and_return_conditional_losses_336142inputs_0inputs_1inputs_2"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
(__inference_dense_6_layer_call_fn_336151inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_6_layer_call_and_return_conditional_losses_336162inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
(__inference_dense_7_layer_call_fn_336171inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_7_layer_call_and_return_conditional_losses_336182inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
мBй
(__inference_dense_8_layer_call_fn_336191inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
їBє
C__inference_dense_8_layer_call_and_return_conditional_losses_336202inputs"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
Щ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20Bє
9__inference_transform_features_layer_layer_call_fn_335458Building DimensionBuilding_FencedBuilding_PaintedBuilding_TypeCustomer IdGardenGeo_CodeInsured_PeriodNumberOfWindows
Settlement
"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗ	capture_0zИ	capture_1zЙ	capture_2zК	capture_3zЛ	capture_4zМ	capture_5zН	capture_6zО	capture_7zП	capture_9zР
capture_10zС
capture_11zТ
capture_12zУ
capture_14zФ
capture_15zХ
capture_16zЦ
capture_17zЧ
capture_19zШ
capture_20

З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20BК
9__inference_transform_features_layer_layer_call_fn_336268inputs_building_dimensioninputs_building_fencedinputs_building_paintedinputs_building_typeinputs_customer_idinputs_gardeninputs_geo_codeinputs_insured_periodinputs_numberofwindowsinputs_settlement
"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗ	capture_0zИ	capture_1zЙ	capture_2zК	capture_3zЛ	capture_4zМ	capture_5zН	capture_6zО	capture_7zП	capture_9zР
capture_10zС
capture_11zТ
capture_12zУ
capture_14zФ
capture_15zХ
capture_16zЦ
capture_17zЧ
capture_19zШ
capture_20
Њ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20Bе
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_336350inputs_building_dimensioninputs_building_fencedinputs_building_paintedinputs_building_typeinputs_customer_idinputs_gardeninputs_geo_codeinputs_insured_periodinputs_numberofwindowsinputs_settlement
"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗ	capture_0zИ	capture_1zЙ	capture_2zК	capture_3zЛ	capture_4zМ	capture_5zН	capture_6zО	capture_7zП	capture_9zР
capture_10zС
capture_11zТ
capture_12zУ
capture_14zФ
capture_15zХ
capture_16zЦ
capture_17zЧ
capture_19zШ
capture_20
ф
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20B
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_335606Building DimensionBuilding_FencedBuilding_PaintedBuilding_TypeCustomer IdGardenGeo_CodeInsured_PeriodNumberOfWindows
Settlement
"Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗ	capture_0zИ	capture_1zЙ	capture_2zК	capture_3zЛ	capture_4zМ	capture_5zН	capture_6zО	capture_7zП	capture_9zР
capture_10zС
capture_11zТ
capture_12zУ
capture_14zФ
capture_15zХ
capture_16zЦ
capture_17zЧ
capture_19zШ
capture_20
Ш
Щcreated_variables
Ъ	resources
Ыtrackable_objects
Ьinitializers
Эassets
Ю
signatures
$Я_self_saveable_object_factories
transform_fn"
_generic_user_object
п
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20B
__inference_pruned_331410inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10zЗ	capture_0zИ	capture_1zЙ	capture_2zК	capture_3zЛ	capture_4zМ	capture_5zН	capture_6zО	capture_7zП	capture_9zР
capture_10zС
capture_11zТ
capture_12zУ
capture_14zФ
capture_15zХ
capture_16zЦ
capture_17zЧ
capture_19zШ
capture_20
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
R
а	variables
б	keras_api

вtotal

гcount"
_tf_keras_metric
c
д	variables
е	keras_api

жtotal

зcount
и
_fn_kwargs"
_tf_keras_metric
%:#2Adam/m/dense_6/kernel
%:#2Adam/v/dense_6/kernel
:2Adam/m/dense_6/bias
:2Adam/v/dense_6/bias
%:#H2Adam/m/dense_7/kernel
%:#H2Adam/v/dense_7/kernel
:H2Adam/m/dense_7/bias
:H2Adam/v/dense_7/bias
%:#H2Adam/m/dense_8/kernel
%:#H2Adam/v/dense_8/kernel
:2Adam/m/dense_8/bias
:2Adam/v/dense_8/bias
јBѕ
#__inference__update_step_xla_336063gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_336068gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_336073gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_336078gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_336083gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
јBѕ
#__inference__update_step_xla_336088gradientvariable"З
ЎВЊ
FullArgSpec2
args*'
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
й0
к1
л2
м3
н4
о5"
trackable_list_wrapper
 "
trackable_list_wrapper
8
п0
р1
с2"
trackable_list_wrapper
8
т0
у1
ф2"
trackable_list_wrapper
-
хserving_default"
signature_map
 "
trackable_dict_wrapper
0
в0
г1"
trackable_list_wrapper
.
а	variables"
_generic_user_object
:  (2total
:  (2count
0
ж0
з1"
trackable_list_wrapper
.
д	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
V
п_initializer
ц_create_resource
ч_initialize
ш_destroy_resourceR 
V
п_initializer
щ_create_resource
ъ_initialize
ы_destroy_resourceR 
V
р_initializer
ь_create_resource
э_initialize
ю_destroy_resourceR 
V
р_initializer
я_create_resource
№_initialize
ё_destroy_resourceR 
V
с_initializer
ђ_create_resource
ѓ_initialize
є_destroy_resourceR 
V
с_initializer
ѕ_create_resource
і_initialize
ї_destroy_resourceR 
T
т	_filename
$ј_self_saveable_object_factories"
_generic_user_object
T
у	_filename
$љ_self_saveable_object_factories"
_generic_user_object
T
ф	_filename
$њ_self_saveable_object_factories"
_generic_user_object
*
*
* 
џ
З	capture_0
И	capture_1
Й	capture_2
К	capture_3
Л	capture_4
М	capture_5
Н	capture_6
О	capture_7
П	capture_9
Р
capture_10
С
capture_11
Т
capture_12
У
capture_14
Ф
capture_15
Х
capture_16
Ц
capture_17
Ч
capture_19
Ш
capture_20BЊ
$__inference_signature_wrapper_331458inputsinputs_1	inputs_10inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗ	capture_0zИ	capture_1zЙ	capture_2zК	capture_3zЛ	capture_4zМ	capture_5zН	capture_6zО	capture_7zП	capture_9zР
capture_10zС
capture_11zТ
capture_12zУ
capture_14zФ
capture_15zХ
capture_16zЦ
capture_17zЧ
capture_19zШ
capture_20
Ю
ћtrace_02Џ
__inference__creator_336359
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zћtrace_0
в
ќtrace_02Г
__inference__initializer_336377
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zќtrace_0
а
§trace_02Б
__inference__destroyer_336388
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ z§trace_0
Ю
ўtrace_02Џ
__inference__creator_336398
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zўtrace_0
в
џtrace_02Г
__inference__initializer_336416
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zџtrace_0
а
trace_02Б
__inference__destroyer_336427
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
Ю
trace_02Џ
__inference__creator_336436
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
в
trace_02Г
__inference__initializer_336454
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
а
trace_02Б
__inference__destroyer_336465
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
Ю
trace_02Џ
__inference__creator_336475
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
в
trace_02Г
__inference__initializer_336493
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
а
trace_02Б
__inference__destroyer_336504
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
Ю
trace_02Џ
__inference__creator_336513
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
в
trace_02Г
__inference__initializer_336531
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
а
trace_02Б
__inference__destroyer_336542
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
Ю
trace_02Џ
__inference__creator_336552
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
в
trace_02Г
__inference__initializer_336570
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
а
trace_02Б
__inference__destroyer_336581
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ ztrace_0
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
ВBЏ
__inference__creator_336359"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ж
т	capture_0BГ
__inference__initializer_336377"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zт	capture_0
ДBБ
__inference__destroyer_336388"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ВBЏ
__inference__creator_336398"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ж
т	capture_0BГ
__inference__initializer_336416"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zт	capture_0
ДBБ
__inference__destroyer_336427"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ВBЏ
__inference__creator_336436"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ж
у	capture_0BГ
__inference__initializer_336454"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zу	capture_0
ДBБ
__inference__destroyer_336465"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ВBЏ
__inference__creator_336475"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ж
у	capture_0BГ
__inference__initializer_336493"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zу	capture_0
ДBБ
__inference__destroyer_336504"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ВBЏ
__inference__creator_336513"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ж
ф	capture_0BГ
__inference__initializer_336531"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zф	capture_0
ДBБ
__inference__destroyer_336542"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ВBЏ
__inference__creator_336552"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ж
ф	capture_0BГ
__inference__initializer_336570"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ zф	capture_0
ДBБ
__inference__destroyer_336581"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ @
__inference__creator_336359!Ђ

Ђ 
Њ "
unknown @
__inference__creator_336398!Ђ

Ђ 
Њ "
unknown @
__inference__creator_336436!Ђ

Ђ 
Њ "
unknown @
__inference__creator_336475!Ђ

Ђ 
Њ "
unknown @
__inference__creator_336513!Ђ

Ђ 
Њ "
unknown @
__inference__creator_336552!Ђ

Ђ 
Њ "
unknown B
__inference__destroyer_336388!Ђ

Ђ 
Њ "
unknown B
__inference__destroyer_336427!Ђ

Ђ 
Њ "
unknown B
__inference__destroyer_336465!Ђ

Ђ 
Њ "
unknown B
__inference__destroyer_336504!Ђ

Ђ 
Њ "
unknown B
__inference__destroyer_336542!Ђ

Ђ 
Њ "
unknown B
__inference__destroyer_336581!Ђ

Ђ 
Њ "
unknown J
__inference__initializer_336377'тйЂ

Ђ 
Њ "
unknown J
__inference__initializer_336416'тйЂ

Ђ 
Њ "
unknown J
__inference__initializer_336454'улЂ

Ђ 
Њ "
unknown J
__inference__initializer_336493'улЂ

Ђ 
Њ "
unknown J
__inference__initializer_336531'фнЂ

Ђ 
Њ "
unknown J
__inference__initializer_336570'фнЂ

Ђ 
Њ "
unknown 
#__inference__update_step_xla_336063nhЂe
^Ђ[

gradient
41	Ђ
њ

p
` VariableSpec 
`РтдБР?
Њ "
 
#__inference__update_step_xla_336068f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
`ршдБР?
Њ "
 
#__inference__update_step_xla_336073nhЂe
^Ђ[

gradientH
41	Ђ
њH

p
` VariableSpec 
`рй№БР?
Њ "
 
#__inference__update_step_xla_336078f`Ђ]
VЂS

gradientH
0-	Ђ
њH

p
` VariableSpec 
` шБР?
Њ "
 
#__inference__update_step_xla_336083nhЂe
^Ђ[

gradientH
41	Ђ
њH

p
` VariableSpec 
`РндБР?
Њ "
 
#__inference__update_step_xla_336088f`Ђ]
VЂS

gradient
0-	Ђ
њ

p
` VariableSpec 
` ыдБР?
Њ "
 
!__inference__wrapped_model_335308х67>?FGЇЂЃ
Ђ

*'
Building_Type_xfџџџџџџџџџ
+(
Insured_Period_xfџџџџџџџџџ
,)
Building_Fenced_xfџџџџџџџџџ
-*
Building_Painted_xfџџџџџџџџџ
'$
Settlement_xfџџџџџџџџџ
/,
Building Dimension_xfџџџџџџџџџ
Њ "1Њ.
,
dense_8!
dense_8џџџџџџџџџи
I__inference_concatenate_4_layer_call_and_return_conditional_losses_336101ZЂW
PЂM
KH
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 Б
.__inference_concatenate_4_layer_call_fn_336094ZЂW
PЂM
KH
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
Њ "!
unknownџџџџџџџџџќ
I__inference_concatenate_5_layer_call_and_return_conditional_losses_336116Ў~Ђ{
tЂq
ol
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ж
.__inference_concatenate_5_layer_call_fn_336108Ѓ~Ђ{
tЂq
ol
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
Њ "!
unknownџџџџџџџџџГ
I__inference_concatenate_6_layer_call_and_return_conditional_losses_336127f6Ђ3
,Ђ)
'$
"
inputs_0џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
.__inference_concatenate_6_layer_call_fn_336121[6Ђ3
,Ђ)
'$
"
inputs_0џџџџџџџџџ
Њ "!
unknownџџџџџџџџџќ
I__inference_concatenate_7_layer_call_and_return_conditional_losses_336142Ў~Ђ{
tЂq
ol
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ж
.__inference_concatenate_7_layer_call_fn_336134Ѓ~Ђ{
tЂq
ol
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
Њ "!
unknownџџџџџџџџџЊ
C__inference_dense_6_layer_call_and_return_conditional_losses_336162c67/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_dense_6_layer_call_fn_336151X67/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЊ
C__inference_dense_7_layer_call_and_return_conditional_losses_336182c>?/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџH
 
(__inference_dense_7_layer_call_fn_336171X>?/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџHЊ
C__inference_dense_8_layer_call_and_return_conditional_losses_336202cFG/Ђ,
%Ђ"
 
inputsџџџџџџџџџH
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_dense_8_layer_call_fn_336191XFG/Ђ,
%Ђ"
 
inputsџџџџџџџџџH
Њ "!
unknownџџџџџџџџџА
C__inference_model_1_layer_call_and_return_conditional_losses_335908ш67>?FGЏЂЋ
ЃЂ

*'
Building_Type_xfџџџџџџџџџ
+(
Insured_Period_xfџџџџџџџџџ
,)
Building_Fenced_xfџџџџџџџџџ
-*
Building_Painted_xfџџџџџџџџџ
'$
Settlement_xfџџџџџџџџџ
/,
Building Dimension_xfџџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 А
C__inference_model_1_layer_call_and_return_conditional_losses_335936ш67>?FGЏЂЋ
ЃЂ

*'
Building_Type_xfџџџџџџџџџ
+(
Insured_Period_xfџџџџџџџџџ
,)
Building_Fenced_xfџџџџџџџџџ
-*
Building_Painted_xfџџџџџџџџџ
'$
Settlement_xfџџџџџџџџџ
/,
Building Dimension_xfџџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ј
C__inference_model_1_layer_call_and_return_conditional_losses_336020А67>?FGїЂѓ
ыЂч
ми
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 ј
C__inference_model_1_layer_call_and_return_conditional_losses_336058А67>?FGїЂѓ
ыЂч
ми
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
(__inference_model_1_layer_call_fn_335727н67>?FGЏЂЋ
ЃЂ

*'
Building_Type_xfџџџџџџџџџ
+(
Insured_Period_xfџџџџџџџџџ
,)
Building_Fenced_xfџџџџџџџџџ
-*
Building_Painted_xfџџџџџџџџџ
'$
Settlement_xfџџџџџџџџџ
/,
Building Dimension_xfџџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџ
(__inference_model_1_layer_call_fn_335880н67>?FGЏЂЋ
ЃЂ

*'
Building_Type_xfџџџџџџџџџ
+(
Insured_Period_xfџџџџџџџџџ
,)
Building_Fenced_xfџџџџџџџџџ
-*
Building_Painted_xfџџџџџџџџџ
'$
Settlement_xfџџџџџџџџџ
/,
Building Dimension_xfџџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџв
(__inference_model_1_layer_call_fn_335960Ѕ67>?FGїЂѓ
ыЂч
ми
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
p 

 
Њ "!
unknownџџџџџџџџџв
(__inference_model_1_layer_call_fn_335982Ѕ67>?FGїЂѓ
ыЂч
ми
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
p

 
Њ "!
unknownџџџџџџџџџЬ	
__inference_pruned_331410Ў	*ЗИЙКЛМНОйПРСТлУФХЦнЧШЪЂЦ
ОЂК
ЗЊГ
I
Building Dimension30
inputs_building_dimensionџџџџџџџџџ
C
Building_Fenced0-
inputs_building_fencedџџџџџџџџџ
E
Building_Painted1.
inputs_building_paintedџџџџџџџџџ
?
Building_Type.+
inputs_building_typeџџџџџџџџџ	
/
Claim&#
inputs_claimџџџџџџџџџ	
;
Customer Id,)
inputs_customer_idџџџџџџџџџ
1
Garden'$
inputs_gardenџџџџџџџџџ
5
Geo_Code)&
inputs_geo_codeџџџџџџџџџ
A
Insured_Period/,
inputs_insured_periodџџџџџџџџџ
C
NumberOfWindows0-
inputs_numberofwindowsџџџџџџџџџ
9

Settlement+(
inputs_settlementџџџџџџџџџ
Њ "ВЊЎ
D
Building Dimension_xf+(
building_dimension_xfџџџџџџџџџ
B
Building_Fenced_xf,)
building_fenced_xfџџџџџџџџџ
D
Building_Painted_xf-*
building_painted_xfџџџџџџџџџ
:
Building_Type_xf&#
building_type_xfџџџџџџџџџ
(
Claim
claimџџџџџџџџџ	
<
Insured_Period_xf'$
insured_period_xfџџџџџџџџџ
8
Settlement_xf'$
settlement_xfџџџџџџџџџЋ
$__inference_signature_wrapper_331458*ЗИЙКЛМНОйПРСТлУФХЦнЧШЂ
Ђ 
Њ
*
inputs 
inputsџџџџџџџџџ
.
inputs_1"
inputs_1џџџџџџџџџ
0
	inputs_10# 
	inputs_10џџџџџџџџџ
.
inputs_2"
inputs_2џџџџџџџџџ
.
inputs_3"
inputs_3џџџџџџџџџ	
.
inputs_4"
inputs_4џџџџџџџџџ	
.
inputs_5"
inputs_5џџџџџџџџџ
.
inputs_6"
inputs_6џџџџџџџџџ
.
inputs_7"
inputs_7џџџџџџџџџ
.
inputs_8"
inputs_8џџџџџџџџџ
.
inputs_9"
inputs_9џџџџџџџџџ"ВЊЎ
D
Building Dimension_xf+(
building_dimension_xfџџџџџџџџџ
B
Building_Fenced_xf,)
building_fenced_xfџџџџџџџџџ
D
Building_Painted_xf-*
building_painted_xfџџџџџџџџџ
:
Building_Type_xf&#
building_type_xfџџџџџџџџџ
(
Claim
claimџџџџџџџџџ	
<
Insured_Period_xf'$
insured_period_xfџџџџџџџџџ
8
Settlement_xf'$
settlement_xfџџџџџџџџџЫ
$__inference_signature_wrapper_335269Ђ0ЗИЙКЛМНОйПРСТлУФХЦнЧШ67>?FG9Ђ6
Ђ 
/Њ,
*
examples
examplesџџџџџџџџџ"3Њ0
.
output_0"
output_0џџџџџџџџџЈ	
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_335606Я*ЗИЙКЛМНОйПРСТлУФХЦнЧШгЂЯ
ЧЂУ
РЊМ
B
Building Dimension,)
Building Dimensionџџџџџџџџџ
<
Building_Fenced)&
Building_Fencedџџџџџџџџџ
>
Building_Painted*'
Building_Paintedџџџџџџџџџ
8
Building_Type'$
Building_Typeџџџџџџџџџ	
4
Customer Id%"
Customer Idџџџџџџџџџ
*
Garden 
Gardenџџџџџџџџџ
.
Geo_Code"
Geo_Codeџџџџџџџџџ
:
Insured_Period(%
Insured_Periodџџџџџџџџџ
<
NumberOfWindows)&
NumberOfWindowsџџџџџџџџџ
2

Settlement$!

Settlementџџџџџџџџџ
Њ "ЪЂЦ
ОЊК
M
Building Dimension_xf41
tensor_0_building_dimension_xfџџџџџџџџџ
K
Building_Fenced_xf52
tensor_0_building_fenced_xfџџџџџџџџџ
M
Building_Painted_xf63
tensor_0_building_painted_xfџџџџџџџџџ
C
Building_Type_xf/,
tensor_0_building_type_xfџџџџџџџџџ
E
Insured_Period_xf0-
tensor_0_insured_period_xfџџџџџџџџџ
A
Settlement_xf0-
tensor_0_settlement_xfџџџџџџџџџ
 ю	
T__inference_transform_features_layer_layer_call_and_return_conditional_losses_336350	*ЗИЙКЛМНОйПРСТлУФХЦнЧШЂ
Ђ
Њ
I
Building Dimension30
inputs_building_dimensionџџџџџџџџџ
C
Building_Fenced0-
inputs_building_fencedџџџџџџџџџ
E
Building_Painted1.
inputs_building_paintedџџџџџџџџџ
?
Building_Type.+
inputs_building_typeџџџџџџџџџ	
;
Customer Id,)
inputs_customer_idџџџџџџџџџ
1
Garden'$
inputs_gardenџџџџџџџџџ
5
Geo_Code)&
inputs_geo_codeџџџџџџџџџ
A
Insured_Period/,
inputs_insured_periodџџџџџџџџџ
C
NumberOfWindows0-
inputs_numberofwindowsџџџџџџџџџ
9

Settlement+(
inputs_settlementџџџџџџџџџ
Њ "ЪЂЦ
ОЊК
M
Building Dimension_xf41
tensor_0_building_dimension_xfџџџџџџџџџ
K
Building_Fenced_xf52
tensor_0_building_fenced_xfџџџџџџџџџ
M
Building_Painted_xf63
tensor_0_building_painted_xfџџџџџџџџџ
C
Building_Type_xf/,
tensor_0_building_type_xfџџџџџџџџџ
E
Insured_Period_xf0-
tensor_0_insured_period_xfџџџџџџџџџ
A
Settlement_xf0-
tensor_0_settlement_xfџџџџџџџџџ
 Ы
9__inference_transform_features_layer_layer_call_fn_335458*ЗИЙКЛМНОйПРСТлУФХЦнЧШгЂЯ
ЧЂУ
РЊМ
B
Building Dimension,)
Building Dimensionџџџџџџџџџ
<
Building_Fenced)&
Building_Fencedџџџџџџџџџ
>
Building_Painted*'
Building_Paintedџџџџџџџџџ
8
Building_Type'$
Building_Typeџџџџџџџџџ	
4
Customer Id%"
Customer Idџџџџџџџџџ
*
Garden 
Gardenџџџџџџџџџ
.
Geo_Code"
Geo_Codeџџџџџџџџџ
:
Insured_Period(%
Insured_Periodџџџџџџџџџ
<
NumberOfWindows)&
NumberOfWindowsџџџџџџџџџ
2

Settlement$!

Settlementџџџџџџџџџ
Њ "Њ
D
Building Dimension_xf+(
building_dimension_xfџџџџџџџџџ
B
Building_Fenced_xf,)
building_fenced_xfџџџџџџџџџ
D
Building_Painted_xf-*
building_painted_xfџџџџџџџџџ
:
Building_Type_xf&#
building_type_xfџџџџџџџџџ
<
Insured_Period_xf'$
insured_period_xfџџџџџџџџџ
8
Settlement_xf'$
settlement_xfџџџџџџџџџ	
9__inference_transform_features_layer_layer_call_fn_336268г*ЗИЙКЛМНОйПРСТлУФХЦнЧШЂ
Ђ
Њ
I
Building Dimension30
inputs_building_dimensionџџџџџџџџџ
C
Building_Fenced0-
inputs_building_fencedџџџџџџџџџ
E
Building_Painted1.
inputs_building_paintedџџџџџџџџџ
?
Building_Type.+
inputs_building_typeџџџџџџџџџ	
;
Customer Id,)
inputs_customer_idџџџџџџџџџ
1
Garden'$
inputs_gardenџџџџџџџџџ
5
Geo_Code)&
inputs_geo_codeџџџџџџџџџ
A
Insured_Period/,
inputs_insured_periodџџџџџџџџџ
C
NumberOfWindows0-
inputs_numberofwindowsџџџџџџџџџ
9

Settlement+(
inputs_settlementџџџџџџџџџ
Њ "Њ
D
Building Dimension_xf+(
building_dimension_xfџџџџџџџџџ
B
Building_Fenced_xf,)
building_fenced_xfџџџџџџџџџ
D
Building_Painted_xf-*
building_painted_xfџџџџџџџџџ
:
Building_Type_xf&#
building_type_xfџџџџџџџџџ
<
Insured_Period_xf'$
insured_period_xfџџџџџџџџџ
8
Settlement_xf'$
settlement_xfџџџџџџџџџ