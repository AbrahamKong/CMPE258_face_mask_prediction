??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
_
Pad

input"T
paddings"	Tpaddings
output"T"	
Ttype"
	Tpaddingstype0:
2	
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.8.02v2.8.0-0-g3f878cff5b68??
?
conv2d_214/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_214/kernel

%conv2d_214/kernel/Read/ReadVariableOpReadVariableOpconv2d_214/kernel*&
_output_shapes
:@*
dtype0
?
conv2d_215/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*"
shared_nameconv2d_215/kernel
?
%conv2d_215/kernel/Read/ReadVariableOpReadVariableOpconv2d_215/kernel*'
_output_shapes
:@?*
dtype0
?
$batch_instance_normalization_169/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$batch_instance_normalization_169/rho
?
8batch_instance_normalization_169/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_169/rho*
_output_shapes	
:?*
dtype0
?
&batch_instance_normalization_169/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_instance_normalization_169/gamma
?
:batch_instance_normalization_169/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_169/gamma*
_output_shapes	
:?*
dtype0
?
%batch_instance_normalization_169/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_instance_normalization_169/beta
?
9batch_instance_normalization_169/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_169/beta*
_output_shapes	
:?*
dtype0
?
conv2d_216/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*"
shared_nameconv2d_216/kernel
?
%conv2d_216/kernel/Read/ReadVariableOpReadVariableOpconv2d_216/kernel*(
_output_shapes
:??*
dtype0
?
$batch_instance_normalization_170/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$batch_instance_normalization_170/rho
?
8batch_instance_normalization_170/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_170/rho*
_output_shapes	
:?*
dtype0
?
&batch_instance_normalization_170/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_instance_normalization_170/gamma
?
:batch_instance_normalization_170/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_170/gamma*
_output_shapes	
:?*
dtype0
?
%batch_instance_normalization_170/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_instance_normalization_170/beta
?
9batch_instance_normalization_170/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_170/beta*
_output_shapes	
:?*
dtype0
?
conv2d_217/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*"
shared_nameconv2d_217/kernel
?
%conv2d_217/kernel/Read/ReadVariableOpReadVariableOpconv2d_217/kernel*(
_output_shapes
:??*
dtype0
?
$batch_instance_normalization_171/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$batch_instance_normalization_171/rho
?
8batch_instance_normalization_171/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_171/rho*
_output_shapes	
:?*
dtype0
?
&batch_instance_normalization_171/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_instance_normalization_171/gamma
?
:batch_instance_normalization_171/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_171/gamma*
_output_shapes	
:?*
dtype0
?
%batch_instance_normalization_171/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%batch_instance_normalization_171/beta
?
9batch_instance_normalization_171/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_171/beta*
_output_shapes	
:?*
dtype0
?
conv2d_218/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*"
shared_nameconv2d_218/kernel
?
%conv2d_218/kernel/Read/ReadVariableOpReadVariableOpconv2d_218/kernel*'
_output_shapes
:?*
dtype0
v
conv2d_218/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_218/bias
o
#conv2d_218/bias/Read/ReadVariableOpReadVariableOpconv2d_218/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?:
value?:B?: B?:
?
conv1_1
conv2_1
	bn2_1
conv3_1
	bn3_1
	zero_pad1
conv4_1
	bn4_1
		zero_pad2

conv5_1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses*
?
!rho
	"gamma
#beta
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
?

*kernel
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*
?
1rho
	2gamma
3beta
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses*
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 
?

@kernel
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses*
?
Grho
	Hgamma
Ibeta
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses*
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
?

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses*
r
0
1
!2
"3
#4
*5
16
27
38
@9
G10
H11
I12
V13
W14*
r
0
1
!2
"3
#4
*5
16
27
38
@9
G10
H11
I12
V13
W14*
* 
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

cserving_default* 
TN
VARIABLE_VALUEconv2d_214/kernel)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_215/kernel)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_169/rho$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_169/gamma&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_169/beta%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

!0
"1
#2*

!0
"1
#2*
* 
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_216/kernel)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

*0*

*0*
* 
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_170/rho$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_170/gamma&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_170/beta%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

10
21
32*

10
21
32*
* 
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 
* 
* 
TN
VARIABLE_VALUEconv2d_217/kernel)conv4_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

@0*

@0*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_171/rho$bn4_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_171/gamma&bn4_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_171/beta%bn4_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1
I2*

G0
H1
I2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 
* 
* 
TN
VARIABLE_VALUEconv2d_218/kernel)conv5_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_218/bias'conv5_1/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses*
* 
* 
* 
J
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
9*
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
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_214/kernelconv2d_215/kernel$batch_instance_normalization_169/rho&batch_instance_normalization_169/gamma%batch_instance_normalization_169/betaconv2d_216/kernel$batch_instance_normalization_170/rho&batch_instance_normalization_170/gamma%batch_instance_normalization_170/betaconv2d_217/kernel$batch_instance_normalization_171/rho&batch_instance_normalization_171/gamma%batch_instance_normalization_171/betaconv2d_218/kernelconv2d_218/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_56547638
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_214/kernel/Read/ReadVariableOp%conv2d_215/kernel/Read/ReadVariableOp8batch_instance_normalization_169/rho/Read/ReadVariableOp:batch_instance_normalization_169/gamma/Read/ReadVariableOp9batch_instance_normalization_169/beta/Read/ReadVariableOp%conv2d_216/kernel/Read/ReadVariableOp8batch_instance_normalization_170/rho/Read/ReadVariableOp:batch_instance_normalization_170/gamma/Read/ReadVariableOp9batch_instance_normalization_170/beta/Read/ReadVariableOp%conv2d_217/kernel/Read/ReadVariableOp8batch_instance_normalization_171/rho/Read/ReadVariableOp:batch_instance_normalization_171/gamma/Read/ReadVariableOp9batch_instance_normalization_171/beta/Read/ReadVariableOp%conv2d_218/kernel/Read/ReadVariableOp#conv2d_218/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8? **
f%R#
!__inference__traced_save_56547956
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_214/kernelconv2d_215/kernel$batch_instance_normalization_169/rho&batch_instance_normalization_169/gamma%batch_instance_normalization_169/betaconv2d_216/kernel$batch_instance_normalization_170/rho&batch_instance_normalization_170/gamma%batch_instance_normalization_170/betaconv2d_217/kernel$batch_instance_normalization_171/rho&batch_instance_normalization_171/gamma%batch_instance_normalization_171/betaconv2d_218/kernelconv2d_218/bias*
Tin
2*
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
GPU2*0J 8? *-
f(R&
$__inference__traced_restore_56548011??
?
?
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56547666

inputs9
conv2d_readvariableop_resource:@?
identity??Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?<
?	
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56546895

inputs-
conv2d_214_56546692:@.
conv2d_215_56546704:@?8
)batch_instance_normalization_169_56546748:	?8
)batch_instance_normalization_169_56546750:	?8
)batch_instance_normalization_169_56546752:	?/
conv2d_216_56546764:??8
)batch_instance_normalization_170_56546808:	?8
)batch_instance_normalization_170_56546810:	?8
)batch_instance_normalization_170_56546812:	?/
conv2d_217_56546825:??8
)batch_instance_normalization_171_56546869:	?8
)batch_instance_normalization_171_56546871:	?8
)batch_instance_normalization_171_56546873:	?.
conv2d_218_56546889:?!
conv2d_218_56546891:
identity??8batch_instance_normalization_169/StatefulPartitionedCall?8batch_instance_normalization_170/StatefulPartitionedCall?8batch_instance_normalization_171/StatefulPartitionedCall?"conv2d_214/StatefulPartitionedCall?"conv2d_215/StatefulPartitionedCall?"conv2d_216/StatefulPartitionedCall?"conv2d_217/StatefulPartitionedCall?"conv2d_218/StatefulPartitionedCall?
"conv2d_214/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_214_56546692*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56546691t
	LeakyRelu	LeakyRelu+conv2d_214/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@@@?
"conv2d_215/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0conv2d_215_56546704*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56546703?
8batch_instance_normalization_169/StatefulPartitionedCallStatefulPartitionedCall+conv2d_215/StatefulPartitionedCall:output:0)batch_instance_normalization_169_56546748)batch_instance_normalization_169_56546750)batch_instance_normalization_169_56546752*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56546747?
LeakyRelu_1	LeakyReluAbatch_instance_normalization_169/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????  ??
"conv2d_216/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_216_56546764*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56546763?
8batch_instance_normalization_170/StatefulPartitionedCallStatefulPartitionedCall+conv2d_216/StatefulPartitionedCall:output:0)batch_instance_normalization_170_56546808)batch_instance_normalization_170_56546810)batch_instance_normalization_170_56546812*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56546807?
LeakyRelu_2	LeakyReluAbatch_instance_normalization_170/StatefulPartitionedCall:output:0*0
_output_shapes
:???????????
!zero_padding2d_16/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56546661?
"conv2d_217/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_16/PartitionedCall:output:0conv2d_217_56546825*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56546824?
8batch_instance_normalization_171/StatefulPartitionedCallStatefulPartitionedCall+conv2d_217/StatefulPartitionedCall:output:0)batch_instance_normalization_171_56546869)batch_instance_normalization_171_56546871)batch_instance_normalization_171_56546873*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56546868?
LeakyRelu_3	LeakyReluAbatch_instance_normalization_171/StatefulPartitionedCall:output:0*0
_output_shapes
:???????????
!zero_padding2d_17/PartitionedCallPartitionedCallLeakyRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56546674?
"conv2d_218/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_17/PartitionedCall:output:0conv2d_218_56546889conv2d_218_56546891*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56546888?
IdentityIdentity+conv2d_218/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp9^batch_instance_normalization_169/StatefulPartitionedCall9^batch_instance_normalization_170/StatefulPartitionedCall9^batch_instance_normalization_171/StatefulPartitionedCall#^conv2d_214/StatefulPartitionedCall#^conv2d_215/StatefulPartitionedCall#^conv2d_216/StatefulPartitionedCall#^conv2d_217/StatefulPartitionedCall#^conv2d_218/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 2t
8batch_instance_normalization_169/StatefulPartitionedCall8batch_instance_normalization_169/StatefulPartitionedCall2t
8batch_instance_normalization_170/StatefulPartitionedCall8batch_instance_normalization_170/StatefulPartitionedCall2t
8batch_instance_normalization_171/StatefulPartitionedCall8batch_instance_normalization_171/StatefulPartitionedCall2H
"conv2d_214/StatefulPartitionedCall"conv2d_214/StatefulPartitionedCall2H
"conv2d_215/StatefulPartitionedCall"conv2d_215/StatefulPartitionedCall2H
"conv2d_216/StatefulPartitionedCall"conv2d_216/StatefulPartitionedCall2H
"conv2d_217/StatefulPartitionedCall"conv2d_217/StatefulPartitionedCall2H
"conv2d_218/StatefulPartitionedCall"conv2d_218/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_216_layer_call_fn_56547724

inputs#
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56546763x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
1__inference_face_d_whole_4_layer_call_fn_56546928
input_1!
unknown:@$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?

unknown_10:	?

unknown_11:	?%

unknown_12:?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56546895w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
1__inference_face_d_whole_4_layer_call_fn_56547290

inputs!
unknown:@$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?

unknown_10:	?

unknown_11:	?%

unknown_12:?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56546895w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?<
?	
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547091

inputs-
conv2d_214_56547046:@.
conv2d_215_56547050:@?8
)batch_instance_normalization_169_56547053:	?8
)batch_instance_normalization_169_56547055:	?8
)batch_instance_normalization_169_56547057:	?/
conv2d_216_56547061:??8
)batch_instance_normalization_170_56547064:	?8
)batch_instance_normalization_170_56547066:	?8
)batch_instance_normalization_170_56547068:	?/
conv2d_217_56547073:??8
)batch_instance_normalization_171_56547076:	?8
)batch_instance_normalization_171_56547078:	?8
)batch_instance_normalization_171_56547080:	?.
conv2d_218_56547085:?!
conv2d_218_56547087:
identity??8batch_instance_normalization_169/StatefulPartitionedCall?8batch_instance_normalization_170/StatefulPartitionedCall?8batch_instance_normalization_171/StatefulPartitionedCall?"conv2d_214/StatefulPartitionedCall?"conv2d_215/StatefulPartitionedCall?"conv2d_216/StatefulPartitionedCall?"conv2d_217/StatefulPartitionedCall?"conv2d_218/StatefulPartitionedCall?
"conv2d_214/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_214_56547046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56546691t
	LeakyRelu	LeakyRelu+conv2d_214/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@@@?
"conv2d_215/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0conv2d_215_56547050*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56546703?
8batch_instance_normalization_169/StatefulPartitionedCallStatefulPartitionedCall+conv2d_215/StatefulPartitionedCall:output:0)batch_instance_normalization_169_56547053)batch_instance_normalization_169_56547055)batch_instance_normalization_169_56547057*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56546747?
LeakyRelu_1	LeakyReluAbatch_instance_normalization_169/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????  ??
"conv2d_216/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_216_56547061*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56546763?
8batch_instance_normalization_170/StatefulPartitionedCallStatefulPartitionedCall+conv2d_216/StatefulPartitionedCall:output:0)batch_instance_normalization_170_56547064)batch_instance_normalization_170_56547066)batch_instance_normalization_170_56547068*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56546807?
LeakyRelu_2	LeakyReluAbatch_instance_normalization_170/StatefulPartitionedCall:output:0*0
_output_shapes
:???????????
!zero_padding2d_16/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56546661?
"conv2d_217/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_16/PartitionedCall:output:0conv2d_217_56547073*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56546824?
8batch_instance_normalization_171/StatefulPartitionedCallStatefulPartitionedCall+conv2d_217/StatefulPartitionedCall:output:0)batch_instance_normalization_171_56547076)batch_instance_normalization_171_56547078)batch_instance_normalization_171_56547080*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56546868?
LeakyRelu_3	LeakyReluAbatch_instance_normalization_171/StatefulPartitionedCall:output:0*0
_output_shapes
:???????????
!zero_padding2d_17/PartitionedCallPartitionedCallLeakyRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56546674?
"conv2d_218/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_17/PartitionedCall:output:0conv2d_218_56547085conv2d_218_56547087*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56546888?
IdentityIdentity+conv2d_218/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp9^batch_instance_normalization_169/StatefulPartitionedCall9^batch_instance_normalization_170/StatefulPartitionedCall9^batch_instance_normalization_171/StatefulPartitionedCall#^conv2d_214/StatefulPartitionedCall#^conv2d_215/StatefulPartitionedCall#^conv2d_216/StatefulPartitionedCall#^conv2d_217/StatefulPartitionedCall#^conv2d_218/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 2t
8batch_instance_normalization_169/StatefulPartitionedCall8batch_instance_normalization_169/StatefulPartitionedCall2t
8batch_instance_normalization_170/StatefulPartitionedCall8batch_instance_normalization_170/StatefulPartitionedCall2t
8batch_instance_normalization_171/StatefulPartitionedCall8batch_instance_normalization_171/StatefulPartitionedCall2H
"conv2d_214/StatefulPartitionedCall"conv2d_214/StatefulPartitionedCall2H
"conv2d_215/StatefulPartitionedCall"conv2d_215/StatefulPartitionedCall2H
"conv2d_216/StatefulPartitionedCall"conv2d_216/StatefulPartitionedCall2H
"conv2d_217/StatefulPartitionedCall"conv2d_217/StatefulPartitionedCall2H
"conv2d_218/StatefulPartitionedCall"conv2d_218/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_218_layer_call_fn_56547878

inputs"
unknown:?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56546888w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56547782
x&
readvariableop_resource:	?,
mul_4_readvariableop_resource:	?,
add_3_readvariableop_resource:	?
identity??ReadVariableOp?ReadVariableOp_1?add_3/ReadVariableOp?mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:??
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:??????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:?I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:?Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:??????????q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:???????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:??????????u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:??????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:??????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:??????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:??????????c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:??????????e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:?]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:??????????_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:??????????o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:?*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:??????????

_user_specified_namex
?
k
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56546674

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?$
?
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56546868
x&
readvariableop_resource:	?,
mul_4_readvariableop_resource:	?,
add_3_readvariableop_resource:	?
identity??ReadVariableOp?ReadVariableOp_1?add_3/ReadVariableOp?mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:??
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:??????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:?I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:?Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:??????????q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:???????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:??????????u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:??????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:??????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:??????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:??????????c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:??????????e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:?]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:??????????_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:??????????o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:?*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:??????????

_user_specified_namex
?
P
4__inference_zero_padding2d_16_layer_call_fn_56547787

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56546661?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
P
4__inference_zero_padding2d_17_layer_call_fn_56547863

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56546674?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
k
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56547869

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56547652

inputs8
conv2d_readvariableop_resource:@
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56547807

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_56547638
input_1!
unknown:@$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?

unknown_10:	?

unknown_11:	?%

unknown_12:?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_56546651w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
1__inference_face_d_whole_4_layer_call_fn_56547325

inputs!
unknown:@$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?

unknown_10:	?

unknown_11:	?%

unknown_12:?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547091w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
k
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56546661

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?$
?
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56546747
x&
readvariableop_resource:	?,
mul_4_readvariableop_resource:	?,
add_3_readvariableop_resource:	?
identity??ReadVariableOp?ReadVariableOp_1?add_3/ReadVariableOp?mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:??
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ?w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:?????????  ?J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:?I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:?Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:?????????  ?q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:???????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ?u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????  ?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:??????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:??????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  ?c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:?????????  ?e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:?]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:?????????  ?_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????  ?o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:?*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????  ?: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:?????????  ?

_user_specified_namex
?
?
-__inference_conv2d_214_layer_call_fn_56547645

inputs!
unknown:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56546691w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56546824

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?$
?
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56546807
x&
readvariableop_resource:	?,
mul_4_readvariableop_resource:	?,
add_3_readvariableop_resource:	?
identity??ReadVariableOp?ReadVariableOp_1?add_3/ReadVariableOp?mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:??
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:??????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:?I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:?Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:??????????q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:???????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:??????????u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:??????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:??????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:??????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:??????????c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:??????????e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:?]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:??????????_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:??????????o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:?*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:??????????

_user_specified_namex
?
?
C__inference_batch_instance_normalization_170_layer_call_fn_56547742
x
unknown:	?
	unknown_0:	?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56546807x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:??????????

_user_specified_namex
?
?
C__inference_batch_instance_normalization_169_layer_call_fn_56547677
x
unknown:	?
	unknown_0:	?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56546747x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????  ?: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????  ?

_user_specified_namex
??
?
#__inference__wrapped_model_56546651
input_1R
8face_d_whole_4_conv2d_214_conv2d_readvariableop_resource:@S
8face_d_whole_4_conv2d_215_conv2d_readvariableop_resource:@?V
Gface_d_whole_4_batch_instance_normalization_169_readvariableop_resource:	?\
Mface_d_whole_4_batch_instance_normalization_169_mul_4_readvariableop_resource:	?\
Mface_d_whole_4_batch_instance_normalization_169_add_3_readvariableop_resource:	?T
8face_d_whole_4_conv2d_216_conv2d_readvariableop_resource:??V
Gface_d_whole_4_batch_instance_normalization_170_readvariableop_resource:	?\
Mface_d_whole_4_batch_instance_normalization_170_mul_4_readvariableop_resource:	?\
Mface_d_whole_4_batch_instance_normalization_170_add_3_readvariableop_resource:	?T
8face_d_whole_4_conv2d_217_conv2d_readvariableop_resource:??V
Gface_d_whole_4_batch_instance_normalization_171_readvariableop_resource:	?\
Mface_d_whole_4_batch_instance_normalization_171_mul_4_readvariableop_resource:	?\
Mface_d_whole_4_batch_instance_normalization_171_add_3_readvariableop_resource:	?S
8face_d_whole_4_conv2d_218_conv2d_readvariableop_resource:?G
9face_d_whole_4_conv2d_218_biasadd_readvariableop_resource:
identity??>face_d_whole_4/batch_instance_normalization_169/ReadVariableOp?@face_d_whole_4/batch_instance_normalization_169/ReadVariableOp_1?Dface_d_whole_4/batch_instance_normalization_169/add_3/ReadVariableOp?Dface_d_whole_4/batch_instance_normalization_169/mul_4/ReadVariableOp?>face_d_whole_4/batch_instance_normalization_170/ReadVariableOp?@face_d_whole_4/batch_instance_normalization_170/ReadVariableOp_1?Dface_d_whole_4/batch_instance_normalization_170/add_3/ReadVariableOp?Dface_d_whole_4/batch_instance_normalization_170/mul_4/ReadVariableOp?>face_d_whole_4/batch_instance_normalization_171/ReadVariableOp?@face_d_whole_4/batch_instance_normalization_171/ReadVariableOp_1?Dface_d_whole_4/batch_instance_normalization_171/add_3/ReadVariableOp?Dface_d_whole_4/batch_instance_normalization_171/mul_4/ReadVariableOp?/face_d_whole_4/conv2d_214/Conv2D/ReadVariableOp?/face_d_whole_4/conv2d_215/Conv2D/ReadVariableOp?/face_d_whole_4/conv2d_216/Conv2D/ReadVariableOp?/face_d_whole_4/conv2d_217/Conv2D/ReadVariableOp?0face_d_whole_4/conv2d_218/BiasAdd/ReadVariableOp?/face_d_whole_4/conv2d_218/Conv2D/ReadVariableOp?
/face_d_whole_4/conv2d_214/Conv2D/ReadVariableOpReadVariableOp8face_d_whole_4_conv2d_214_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
 face_d_whole_4/conv2d_214/Conv2DConv2Dinput_17face_d_whole_4/conv2d_214/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
face_d_whole_4/LeakyRelu	LeakyRelu)face_d_whole_4/conv2d_214/Conv2D:output:0*/
_output_shapes
:?????????@@@?
/face_d_whole_4/conv2d_215/Conv2D/ReadVariableOpReadVariableOp8face_d_whole_4_conv2d_215_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
 face_d_whole_4/conv2d_215/Conv2DConv2D&face_d_whole_4/LeakyRelu:activations:07face_d_whole_4/conv2d_215/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
Nface_d_whole_4/batch_instance_normalization_169/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
<face_d_whole_4/batch_instance_normalization_169/moments/meanMean)face_d_whole_4/conv2d_215/Conv2D:output:0Wface_d_whole_4/batch_instance_normalization_169/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
Dface_d_whole_4/batch_instance_normalization_169/moments/StopGradientStopGradientEface_d_whole_4/batch_instance_normalization_169/moments/mean:output:0*
T0*'
_output_shapes
:??
Iface_d_whole_4/batch_instance_normalization_169/moments/SquaredDifferenceSquaredDifference)face_d_whole_4/conv2d_215/Conv2D:output:0Mface_d_whole_4/batch_instance_normalization_169/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ??
Rface_d_whole_4/batch_instance_normalization_169/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
@face_d_whole_4/batch_instance_normalization_169/moments/varianceMeanMface_d_whole_4/batch_instance_normalization_169/moments/SquaredDifference:z:0[face_d_whole_4/batch_instance_normalization_169/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
3face_d_whole_4/batch_instance_normalization_169/subSub)face_d_whole_4/conv2d_215/Conv2D:output:0Eface_d_whole_4/batch_instance_normalization_169/moments/mean:output:0*
T0*0
_output_shapes
:?????????  ?z
5face_d_whole_4/batch_instance_normalization_169/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
3face_d_whole_4/batch_instance_normalization_169/addAddV2Iface_d_whole_4/batch_instance_normalization_169/moments/variance:output:0>face_d_whole_4/batch_instance_normalization_169/add/y:output:0*
T0*'
_output_shapes
:??
5face_d_whole_4/batch_instance_normalization_169/RsqrtRsqrt7face_d_whole_4/batch_instance_normalization_169/add:z:0*
T0*'
_output_shapes
:??
3face_d_whole_4/batch_instance_normalization_169/mulMul7face_d_whole_4/batch_instance_normalization_169/sub:z:09face_d_whole_4/batch_instance_normalization_169/Rsqrt:y:0*
T0*0
_output_shapes
:?????????  ??
Pface_d_whole_4/batch_instance_normalization_169/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
>face_d_whole_4/batch_instance_normalization_169/moments_1/meanMean)face_d_whole_4/conv2d_215/Conv2D:output:0Yface_d_whole_4/batch_instance_normalization_169/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
Fface_d_whole_4/batch_instance_normalization_169/moments_1/StopGradientStopGradientGface_d_whole_4/batch_instance_normalization_169/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
Kface_d_whole_4/batch_instance_normalization_169/moments_1/SquaredDifferenceSquaredDifference)face_d_whole_4/conv2d_215/Conv2D:output:0Oface_d_whole_4/batch_instance_normalization_169/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ??
Tface_d_whole_4/batch_instance_normalization_169/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
Bface_d_whole_4/batch_instance_normalization_169/moments_1/varianceMeanOface_d_whole_4/batch_instance_normalization_169/moments_1/SquaredDifference:z:0]face_d_whole_4/batch_instance_normalization_169/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
5face_d_whole_4/batch_instance_normalization_169/sub_1Sub)face_d_whole_4/conv2d_215/Conv2D:output:0Gface_d_whole_4/batch_instance_normalization_169/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????  ?|
7face_d_whole_4/batch_instance_normalization_169/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
5face_d_whole_4/batch_instance_normalization_169/add_1AddV2Kface_d_whole_4/batch_instance_normalization_169/moments_1/variance:output:0@face_d_whole_4/batch_instance_normalization_169/add_1/y:output:0*
T0*0
_output_shapes
:???????????
7face_d_whole_4/batch_instance_normalization_169/Rsqrt_1Rsqrt9face_d_whole_4/batch_instance_normalization_169/add_1:z:0*
T0*0
_output_shapes
:???????????
5face_d_whole_4/batch_instance_normalization_169/mul_1Mul9face_d_whole_4/batch_instance_normalization_169/sub_1:z:0;face_d_whole_4/batch_instance_normalization_169/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  ??
>face_d_whole_4/batch_instance_normalization_169/ReadVariableOpReadVariableOpGface_d_whole_4_batch_instance_normalization_169_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_169/mul_2MulFface_d_whole_4/batch_instance_normalization_169/ReadVariableOp:value:07face_d_whole_4/batch_instance_normalization_169/mul:z:0*
T0*0
_output_shapes
:?????????  ??
@face_d_whole_4/batch_instance_normalization_169/ReadVariableOp_1ReadVariableOpGface_d_whole_4_batch_instance_normalization_169_readvariableop_resource*
_output_shapes	
:?*
dtype0|
7face_d_whole_4/batch_instance_normalization_169/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5face_d_whole_4/batch_instance_normalization_169/sub_2Sub@face_d_whole_4/batch_instance_normalization_169/sub_2/x:output:0Hface_d_whole_4/batch_instance_normalization_169/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
5face_d_whole_4/batch_instance_normalization_169/mul_3Mul9face_d_whole_4/batch_instance_normalization_169/sub_2:z:09face_d_whole_4/batch_instance_normalization_169/mul_1:z:0*
T0*0
_output_shapes
:?????????  ??
5face_d_whole_4/batch_instance_normalization_169/add_2AddV29face_d_whole_4/batch_instance_normalization_169/mul_2:z:09face_d_whole_4/batch_instance_normalization_169/mul_3:z:0*
T0*0
_output_shapes
:?????????  ??
Dface_d_whole_4/batch_instance_normalization_169/mul_4/ReadVariableOpReadVariableOpMface_d_whole_4_batch_instance_normalization_169_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_169/mul_4Mul9face_d_whole_4/batch_instance_normalization_169/add_2:z:0Lface_d_whole_4/batch_instance_normalization_169/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
Dface_d_whole_4/batch_instance_normalization_169/add_3/ReadVariableOpReadVariableOpMface_d_whole_4_batch_instance_normalization_169_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_169/add_3AddV29face_d_whole_4/batch_instance_normalization_169/mul_4:z:0Lface_d_whole_4/batch_instance_normalization_169/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
face_d_whole_4/LeakyRelu_1	LeakyRelu9face_d_whole_4/batch_instance_normalization_169/add_3:z:0*0
_output_shapes
:?????????  ??
/face_d_whole_4/conv2d_216/Conv2D/ReadVariableOpReadVariableOp8face_d_whole_4_conv2d_216_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
 face_d_whole_4/conv2d_216/Conv2DConv2D(face_d_whole_4/LeakyRelu_1:activations:07face_d_whole_4/conv2d_216/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
Nface_d_whole_4/batch_instance_normalization_170/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
<face_d_whole_4/batch_instance_normalization_170/moments/meanMean)face_d_whole_4/conv2d_216/Conv2D:output:0Wface_d_whole_4/batch_instance_normalization_170/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
Dface_d_whole_4/batch_instance_normalization_170/moments/StopGradientStopGradientEface_d_whole_4/batch_instance_normalization_170/moments/mean:output:0*
T0*'
_output_shapes
:??
Iface_d_whole_4/batch_instance_normalization_170/moments/SquaredDifferenceSquaredDifference)face_d_whole_4/conv2d_216/Conv2D:output:0Mface_d_whole_4/batch_instance_normalization_170/moments/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Rface_d_whole_4/batch_instance_normalization_170/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
@face_d_whole_4/batch_instance_normalization_170/moments/varianceMeanMface_d_whole_4/batch_instance_normalization_170/moments/SquaredDifference:z:0[face_d_whole_4/batch_instance_normalization_170/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
3face_d_whole_4/batch_instance_normalization_170/subSub)face_d_whole_4/conv2d_216/Conv2D:output:0Eface_d_whole_4/batch_instance_normalization_170/moments/mean:output:0*
T0*0
_output_shapes
:??????????z
5face_d_whole_4/batch_instance_normalization_170/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
3face_d_whole_4/batch_instance_normalization_170/addAddV2Iface_d_whole_4/batch_instance_normalization_170/moments/variance:output:0>face_d_whole_4/batch_instance_normalization_170/add/y:output:0*
T0*'
_output_shapes
:??
5face_d_whole_4/batch_instance_normalization_170/RsqrtRsqrt7face_d_whole_4/batch_instance_normalization_170/add:z:0*
T0*'
_output_shapes
:??
3face_d_whole_4/batch_instance_normalization_170/mulMul7face_d_whole_4/batch_instance_normalization_170/sub:z:09face_d_whole_4/batch_instance_normalization_170/Rsqrt:y:0*
T0*0
_output_shapes
:???????????
Pface_d_whole_4/batch_instance_normalization_170/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
>face_d_whole_4/batch_instance_normalization_170/moments_1/meanMean)face_d_whole_4/conv2d_216/Conv2D:output:0Yface_d_whole_4/batch_instance_normalization_170/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
Fface_d_whole_4/batch_instance_normalization_170/moments_1/StopGradientStopGradientGface_d_whole_4/batch_instance_normalization_170/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
Kface_d_whole_4/batch_instance_normalization_170/moments_1/SquaredDifferenceSquaredDifference)face_d_whole_4/conv2d_216/Conv2D:output:0Oface_d_whole_4/batch_instance_normalization_170/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Tface_d_whole_4/batch_instance_normalization_170/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
Bface_d_whole_4/batch_instance_normalization_170/moments_1/varianceMeanOface_d_whole_4/batch_instance_normalization_170/moments_1/SquaredDifference:z:0]face_d_whole_4/batch_instance_normalization_170/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
5face_d_whole_4/batch_instance_normalization_170/sub_1Sub)face_d_whole_4/conv2d_216/Conv2D:output:0Gface_d_whole_4/batch_instance_normalization_170/moments_1/mean:output:0*
T0*0
_output_shapes
:??????????|
7face_d_whole_4/batch_instance_normalization_170/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
5face_d_whole_4/batch_instance_normalization_170/add_1AddV2Kface_d_whole_4/batch_instance_normalization_170/moments_1/variance:output:0@face_d_whole_4/batch_instance_normalization_170/add_1/y:output:0*
T0*0
_output_shapes
:???????????
7face_d_whole_4/batch_instance_normalization_170/Rsqrt_1Rsqrt9face_d_whole_4/batch_instance_normalization_170/add_1:z:0*
T0*0
_output_shapes
:???????????
5face_d_whole_4/batch_instance_normalization_170/mul_1Mul9face_d_whole_4/batch_instance_normalization_170/sub_1:z:0;face_d_whole_4/batch_instance_normalization_170/Rsqrt_1:y:0*
T0*0
_output_shapes
:???????????
>face_d_whole_4/batch_instance_normalization_170/ReadVariableOpReadVariableOpGface_d_whole_4_batch_instance_normalization_170_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_170/mul_2MulFface_d_whole_4/batch_instance_normalization_170/ReadVariableOp:value:07face_d_whole_4/batch_instance_normalization_170/mul:z:0*
T0*0
_output_shapes
:???????????
@face_d_whole_4/batch_instance_normalization_170/ReadVariableOp_1ReadVariableOpGface_d_whole_4_batch_instance_normalization_170_readvariableop_resource*
_output_shapes	
:?*
dtype0|
7face_d_whole_4/batch_instance_normalization_170/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5face_d_whole_4/batch_instance_normalization_170/sub_2Sub@face_d_whole_4/batch_instance_normalization_170/sub_2/x:output:0Hface_d_whole_4/batch_instance_normalization_170/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
5face_d_whole_4/batch_instance_normalization_170/mul_3Mul9face_d_whole_4/batch_instance_normalization_170/sub_2:z:09face_d_whole_4/batch_instance_normalization_170/mul_1:z:0*
T0*0
_output_shapes
:???????????
5face_d_whole_4/batch_instance_normalization_170/add_2AddV29face_d_whole_4/batch_instance_normalization_170/mul_2:z:09face_d_whole_4/batch_instance_normalization_170/mul_3:z:0*
T0*0
_output_shapes
:???????????
Dface_d_whole_4/batch_instance_normalization_170/mul_4/ReadVariableOpReadVariableOpMface_d_whole_4_batch_instance_normalization_170_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_170/mul_4Mul9face_d_whole_4/batch_instance_normalization_170/add_2:z:0Lface_d_whole_4/batch_instance_normalization_170/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
Dface_d_whole_4/batch_instance_normalization_170/add_3/ReadVariableOpReadVariableOpMface_d_whole_4_batch_instance_normalization_170_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_170/add_3AddV29face_d_whole_4/batch_instance_normalization_170/mul_4:z:0Lface_d_whole_4/batch_instance_normalization_170/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
face_d_whole_4/LeakyRelu_2	LeakyRelu9face_d_whole_4/batch_instance_normalization_170/add_3:z:0*0
_output_shapes
:???????????
-face_d_whole_4/zero_padding2d_16/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ?
$face_d_whole_4/zero_padding2d_16/PadPad(face_d_whole_4/LeakyRelu_2:activations:06face_d_whole_4/zero_padding2d_16/Pad/paddings:output:0*
T0*0
_output_shapes
:???????????
/face_d_whole_4/conv2d_217/Conv2D/ReadVariableOpReadVariableOp8face_d_whole_4_conv2d_217_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
 face_d_whole_4/conv2d_217/Conv2DConv2D-face_d_whole_4/zero_padding2d_16/Pad:output:07face_d_whole_4/conv2d_217/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
Nface_d_whole_4/batch_instance_normalization_171/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
<face_d_whole_4/batch_instance_normalization_171/moments/meanMean)face_d_whole_4/conv2d_217/Conv2D:output:0Wface_d_whole_4/batch_instance_normalization_171/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
Dface_d_whole_4/batch_instance_normalization_171/moments/StopGradientStopGradientEface_d_whole_4/batch_instance_normalization_171/moments/mean:output:0*
T0*'
_output_shapes
:??
Iface_d_whole_4/batch_instance_normalization_171/moments/SquaredDifferenceSquaredDifference)face_d_whole_4/conv2d_217/Conv2D:output:0Mface_d_whole_4/batch_instance_normalization_171/moments/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Rface_d_whole_4/batch_instance_normalization_171/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
@face_d_whole_4/batch_instance_normalization_171/moments/varianceMeanMface_d_whole_4/batch_instance_normalization_171/moments/SquaredDifference:z:0[face_d_whole_4/batch_instance_normalization_171/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
3face_d_whole_4/batch_instance_normalization_171/subSub)face_d_whole_4/conv2d_217/Conv2D:output:0Eface_d_whole_4/batch_instance_normalization_171/moments/mean:output:0*
T0*0
_output_shapes
:??????????z
5face_d_whole_4/batch_instance_normalization_171/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
3face_d_whole_4/batch_instance_normalization_171/addAddV2Iface_d_whole_4/batch_instance_normalization_171/moments/variance:output:0>face_d_whole_4/batch_instance_normalization_171/add/y:output:0*
T0*'
_output_shapes
:??
5face_d_whole_4/batch_instance_normalization_171/RsqrtRsqrt7face_d_whole_4/batch_instance_normalization_171/add:z:0*
T0*'
_output_shapes
:??
3face_d_whole_4/batch_instance_normalization_171/mulMul7face_d_whole_4/batch_instance_normalization_171/sub:z:09face_d_whole_4/batch_instance_normalization_171/Rsqrt:y:0*
T0*0
_output_shapes
:???????????
Pface_d_whole_4/batch_instance_normalization_171/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
>face_d_whole_4/batch_instance_normalization_171/moments_1/meanMean)face_d_whole_4/conv2d_217/Conv2D:output:0Yface_d_whole_4/batch_instance_normalization_171/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
Fface_d_whole_4/batch_instance_normalization_171/moments_1/StopGradientStopGradientGface_d_whole_4/batch_instance_normalization_171/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
Kface_d_whole_4/batch_instance_normalization_171/moments_1/SquaredDifferenceSquaredDifference)face_d_whole_4/conv2d_217/Conv2D:output:0Oface_d_whole_4/batch_instance_normalization_171/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Tface_d_whole_4/batch_instance_normalization_171/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
Bface_d_whole_4/batch_instance_normalization_171/moments_1/varianceMeanOface_d_whole_4/batch_instance_normalization_171/moments_1/SquaredDifference:z:0]face_d_whole_4/batch_instance_normalization_171/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
5face_d_whole_4/batch_instance_normalization_171/sub_1Sub)face_d_whole_4/conv2d_217/Conv2D:output:0Gface_d_whole_4/batch_instance_normalization_171/moments_1/mean:output:0*
T0*0
_output_shapes
:??????????|
7face_d_whole_4/batch_instance_normalization_171/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
5face_d_whole_4/batch_instance_normalization_171/add_1AddV2Kface_d_whole_4/batch_instance_normalization_171/moments_1/variance:output:0@face_d_whole_4/batch_instance_normalization_171/add_1/y:output:0*
T0*0
_output_shapes
:???????????
7face_d_whole_4/batch_instance_normalization_171/Rsqrt_1Rsqrt9face_d_whole_4/batch_instance_normalization_171/add_1:z:0*
T0*0
_output_shapes
:???????????
5face_d_whole_4/batch_instance_normalization_171/mul_1Mul9face_d_whole_4/batch_instance_normalization_171/sub_1:z:0;face_d_whole_4/batch_instance_normalization_171/Rsqrt_1:y:0*
T0*0
_output_shapes
:???????????
>face_d_whole_4/batch_instance_normalization_171/ReadVariableOpReadVariableOpGface_d_whole_4_batch_instance_normalization_171_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_171/mul_2MulFface_d_whole_4/batch_instance_normalization_171/ReadVariableOp:value:07face_d_whole_4/batch_instance_normalization_171/mul:z:0*
T0*0
_output_shapes
:???????????
@face_d_whole_4/batch_instance_normalization_171/ReadVariableOp_1ReadVariableOpGface_d_whole_4_batch_instance_normalization_171_readvariableop_resource*
_output_shapes	
:?*
dtype0|
7face_d_whole_4/batch_instance_normalization_171/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
5face_d_whole_4/batch_instance_normalization_171/sub_2Sub@face_d_whole_4/batch_instance_normalization_171/sub_2/x:output:0Hface_d_whole_4/batch_instance_normalization_171/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
5face_d_whole_4/batch_instance_normalization_171/mul_3Mul9face_d_whole_4/batch_instance_normalization_171/sub_2:z:09face_d_whole_4/batch_instance_normalization_171/mul_1:z:0*
T0*0
_output_shapes
:???????????
5face_d_whole_4/batch_instance_normalization_171/add_2AddV29face_d_whole_4/batch_instance_normalization_171/mul_2:z:09face_d_whole_4/batch_instance_normalization_171/mul_3:z:0*
T0*0
_output_shapes
:???????????
Dface_d_whole_4/batch_instance_normalization_171/mul_4/ReadVariableOpReadVariableOpMface_d_whole_4_batch_instance_normalization_171_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_171/mul_4Mul9face_d_whole_4/batch_instance_normalization_171/add_2:z:0Lface_d_whole_4/batch_instance_normalization_171/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
Dface_d_whole_4/batch_instance_normalization_171/add_3/ReadVariableOpReadVariableOpMface_d_whole_4_batch_instance_normalization_171_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5face_d_whole_4/batch_instance_normalization_171/add_3AddV29face_d_whole_4/batch_instance_normalization_171/mul_4:z:0Lface_d_whole_4/batch_instance_normalization_171/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
face_d_whole_4/LeakyRelu_3	LeakyRelu9face_d_whole_4/batch_instance_normalization_171/add_3:z:0*0
_output_shapes
:???????????
-face_d_whole_4/zero_padding2d_17/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ?
$face_d_whole_4/zero_padding2d_17/PadPad(face_d_whole_4/LeakyRelu_3:activations:06face_d_whole_4/zero_padding2d_17/Pad/paddings:output:0*
T0*0
_output_shapes
:???????????
/face_d_whole_4/conv2d_218/Conv2D/ReadVariableOpReadVariableOp8face_d_whole_4_conv2d_218_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
 face_d_whole_4/conv2d_218/Conv2DConv2D-face_d_whole_4/zero_padding2d_17/Pad:output:07face_d_whole_4/conv2d_218/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
0face_d_whole_4/conv2d_218/BiasAdd/ReadVariableOpReadVariableOp9face_d_whole_4_conv2d_218_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
!face_d_whole_4/conv2d_218/BiasAddBiasAdd)face_d_whole_4/conv2d_218/Conv2D:output:08face_d_whole_4/conv2d_218/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:??????????
IdentityIdentity*face_d_whole_4/conv2d_218/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????	
NoOpNoOp?^face_d_whole_4/batch_instance_normalization_169/ReadVariableOpA^face_d_whole_4/batch_instance_normalization_169/ReadVariableOp_1E^face_d_whole_4/batch_instance_normalization_169/add_3/ReadVariableOpE^face_d_whole_4/batch_instance_normalization_169/mul_4/ReadVariableOp?^face_d_whole_4/batch_instance_normalization_170/ReadVariableOpA^face_d_whole_4/batch_instance_normalization_170/ReadVariableOp_1E^face_d_whole_4/batch_instance_normalization_170/add_3/ReadVariableOpE^face_d_whole_4/batch_instance_normalization_170/mul_4/ReadVariableOp?^face_d_whole_4/batch_instance_normalization_171/ReadVariableOpA^face_d_whole_4/batch_instance_normalization_171/ReadVariableOp_1E^face_d_whole_4/batch_instance_normalization_171/add_3/ReadVariableOpE^face_d_whole_4/batch_instance_normalization_171/mul_4/ReadVariableOp0^face_d_whole_4/conv2d_214/Conv2D/ReadVariableOp0^face_d_whole_4/conv2d_215/Conv2D/ReadVariableOp0^face_d_whole_4/conv2d_216/Conv2D/ReadVariableOp0^face_d_whole_4/conv2d_217/Conv2D/ReadVariableOp1^face_d_whole_4/conv2d_218/BiasAdd/ReadVariableOp0^face_d_whole_4/conv2d_218/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 2?
>face_d_whole_4/batch_instance_normalization_169/ReadVariableOp>face_d_whole_4/batch_instance_normalization_169/ReadVariableOp2?
@face_d_whole_4/batch_instance_normalization_169/ReadVariableOp_1@face_d_whole_4/batch_instance_normalization_169/ReadVariableOp_12?
Dface_d_whole_4/batch_instance_normalization_169/add_3/ReadVariableOpDface_d_whole_4/batch_instance_normalization_169/add_3/ReadVariableOp2?
Dface_d_whole_4/batch_instance_normalization_169/mul_4/ReadVariableOpDface_d_whole_4/batch_instance_normalization_169/mul_4/ReadVariableOp2?
>face_d_whole_4/batch_instance_normalization_170/ReadVariableOp>face_d_whole_4/batch_instance_normalization_170/ReadVariableOp2?
@face_d_whole_4/batch_instance_normalization_170/ReadVariableOp_1@face_d_whole_4/batch_instance_normalization_170/ReadVariableOp_12?
Dface_d_whole_4/batch_instance_normalization_170/add_3/ReadVariableOpDface_d_whole_4/batch_instance_normalization_170/add_3/ReadVariableOp2?
Dface_d_whole_4/batch_instance_normalization_170/mul_4/ReadVariableOpDface_d_whole_4/batch_instance_normalization_170/mul_4/ReadVariableOp2?
>face_d_whole_4/batch_instance_normalization_171/ReadVariableOp>face_d_whole_4/batch_instance_normalization_171/ReadVariableOp2?
@face_d_whole_4/batch_instance_normalization_171/ReadVariableOp_1@face_d_whole_4/batch_instance_normalization_171/ReadVariableOp_12?
Dface_d_whole_4/batch_instance_normalization_171/add_3/ReadVariableOpDface_d_whole_4/batch_instance_normalization_171/add_3/ReadVariableOp2?
Dface_d_whole_4/batch_instance_normalization_171/mul_4/ReadVariableOpDface_d_whole_4/batch_instance_normalization_171/mul_4/ReadVariableOp2b
/face_d_whole_4/conv2d_214/Conv2D/ReadVariableOp/face_d_whole_4/conv2d_214/Conv2D/ReadVariableOp2b
/face_d_whole_4/conv2d_215/Conv2D/ReadVariableOp/face_d_whole_4/conv2d_215/Conv2D/ReadVariableOp2b
/face_d_whole_4/conv2d_216/Conv2D/ReadVariableOp/face_d_whole_4/conv2d_216/Conv2D/ReadVariableOp2b
/face_d_whole_4/conv2d_217/Conv2D/ReadVariableOp/face_d_whole_4/conv2d_217/Conv2D/ReadVariableOp2d
0face_d_whole_4/conv2d_218/BiasAdd/ReadVariableOp0face_d_whole_4/conv2d_218/BiasAdd/ReadVariableOp2b
/face_d_whole_4/conv2d_218/Conv2D/ReadVariableOp/face_d_whole_4/conv2d_218/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?

?
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56546888

inputs9
conv2d_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_batch_instance_normalization_171_layer_call_fn_56547818
x
unknown:	?
	unknown_0:	?
	unknown_1:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56546868x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:??????????

_user_specified_namex
?
?
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56546691

inputs8
conv2d_readvariableop_resource:@
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
1__inference_face_d_whole_4_layer_call_fn_56547159
input_1!
unknown:@$
	unknown_0:@?
	unknown_1:	?
	unknown_2:	?
	unknown_3:	?%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?

unknown_10:	?

unknown_11:	?%

unknown_12:?

unknown_13:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547091w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
??
?
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547463

inputsC
)conv2d_214_conv2d_readvariableop_resource:@D
)conv2d_215_conv2d_readvariableop_resource:@?G
8batch_instance_normalization_169_readvariableop_resource:	?M
>batch_instance_normalization_169_mul_4_readvariableop_resource:	?M
>batch_instance_normalization_169_add_3_readvariableop_resource:	?E
)conv2d_216_conv2d_readvariableop_resource:??G
8batch_instance_normalization_170_readvariableop_resource:	?M
>batch_instance_normalization_170_mul_4_readvariableop_resource:	?M
>batch_instance_normalization_170_add_3_readvariableop_resource:	?E
)conv2d_217_conv2d_readvariableop_resource:??G
8batch_instance_normalization_171_readvariableop_resource:	?M
>batch_instance_normalization_171_mul_4_readvariableop_resource:	?M
>batch_instance_normalization_171_add_3_readvariableop_resource:	?D
)conv2d_218_conv2d_readvariableop_resource:?8
*conv2d_218_biasadd_readvariableop_resource:
identity??/batch_instance_normalization_169/ReadVariableOp?1batch_instance_normalization_169/ReadVariableOp_1?5batch_instance_normalization_169/add_3/ReadVariableOp?5batch_instance_normalization_169/mul_4/ReadVariableOp?/batch_instance_normalization_170/ReadVariableOp?1batch_instance_normalization_170/ReadVariableOp_1?5batch_instance_normalization_170/add_3/ReadVariableOp?5batch_instance_normalization_170/mul_4/ReadVariableOp?/batch_instance_normalization_171/ReadVariableOp?1batch_instance_normalization_171/ReadVariableOp_1?5batch_instance_normalization_171/add_3/ReadVariableOp?5batch_instance_normalization_171/mul_4/ReadVariableOp? conv2d_214/Conv2D/ReadVariableOp? conv2d_215/Conv2D/ReadVariableOp? conv2d_216/Conv2D/ReadVariableOp? conv2d_217/Conv2D/ReadVariableOp?!conv2d_218/BiasAdd/ReadVariableOp? conv2d_218/Conv2D/ReadVariableOp?
 conv2d_214/Conv2D/ReadVariableOpReadVariableOp)conv2d_214_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_214/Conv2DConv2Dinputs(conv2d_214/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
c
	LeakyRelu	LeakyReluconv2d_214/Conv2D:output:0*/
_output_shapes
:?????????@@@?
 conv2d_215/Conv2D/ReadVariableOpReadVariableOp)conv2d_215_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_215/Conv2DConv2DLeakyRelu:activations:0(conv2d_215/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
?batch_instance_normalization_169/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-batch_instance_normalization_169/moments/meanMeanconv2d_215/Conv2D:output:0Hbatch_instance_normalization_169/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
5batch_instance_normalization_169/moments/StopGradientStopGradient6batch_instance_normalization_169/moments/mean:output:0*
T0*'
_output_shapes
:??
:batch_instance_normalization_169/moments/SquaredDifferenceSquaredDifferenceconv2d_215/Conv2D:output:0>batch_instance_normalization_169/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ??
Cbatch_instance_normalization_169/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1batch_instance_normalization_169/moments/varianceMean>batch_instance_normalization_169/moments/SquaredDifference:z:0Lbatch_instance_normalization_169/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
$batch_instance_normalization_169/subSubconv2d_215/Conv2D:output:06batch_instance_normalization_169/moments/mean:output:0*
T0*0
_output_shapes
:?????????  ?k
&batch_instance_normalization_169/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
$batch_instance_normalization_169/addAddV2:batch_instance_normalization_169/moments/variance:output:0/batch_instance_normalization_169/add/y:output:0*
T0*'
_output_shapes
:??
&batch_instance_normalization_169/RsqrtRsqrt(batch_instance_normalization_169/add:z:0*
T0*'
_output_shapes
:??
$batch_instance_normalization_169/mulMul(batch_instance_normalization_169/sub:z:0*batch_instance_normalization_169/Rsqrt:y:0*
T0*0
_output_shapes
:?????????  ??
Abatch_instance_normalization_169/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
/batch_instance_normalization_169/moments_1/meanMeanconv2d_215/Conv2D:output:0Jbatch_instance_normalization_169/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
7batch_instance_normalization_169/moments_1/StopGradientStopGradient8batch_instance_normalization_169/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
<batch_instance_normalization_169/moments_1/SquaredDifferenceSquaredDifferenceconv2d_215/Conv2D:output:0@batch_instance_normalization_169/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ??
Ebatch_instance_normalization_169/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
3batch_instance_normalization_169/moments_1/varianceMean@batch_instance_normalization_169/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_169/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
&batch_instance_normalization_169/sub_1Subconv2d_215/Conv2D:output:08batch_instance_normalization_169/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????  ?m
(batch_instance_normalization_169/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&batch_instance_normalization_169/add_1AddV2<batch_instance_normalization_169/moments_1/variance:output:01batch_instance_normalization_169/add_1/y:output:0*
T0*0
_output_shapes
:???????????
(batch_instance_normalization_169/Rsqrt_1Rsqrt*batch_instance_normalization_169/add_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_169/mul_1Mul*batch_instance_normalization_169/sub_1:z:0,batch_instance_normalization_169/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  ??
/batch_instance_normalization_169/ReadVariableOpReadVariableOp8batch_instance_normalization_169_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_169/mul_2Mul7batch_instance_normalization_169/ReadVariableOp:value:0(batch_instance_normalization_169/mul:z:0*
T0*0
_output_shapes
:?????????  ??
1batch_instance_normalization_169/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_169_readvariableop_resource*
_output_shapes	
:?*
dtype0m
(batch_instance_normalization_169/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&batch_instance_normalization_169/sub_2Sub1batch_instance_normalization_169/sub_2/x:output:09batch_instance_normalization_169/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
&batch_instance_normalization_169/mul_3Mul*batch_instance_normalization_169/sub_2:z:0*batch_instance_normalization_169/mul_1:z:0*
T0*0
_output_shapes
:?????????  ??
&batch_instance_normalization_169/add_2AddV2*batch_instance_normalization_169/mul_2:z:0*batch_instance_normalization_169/mul_3:z:0*
T0*0
_output_shapes
:?????????  ??
5batch_instance_normalization_169/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_169_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_169/mul_4Mul*batch_instance_normalization_169/add_2:z:0=batch_instance_normalization_169/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
5batch_instance_normalization_169/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_169_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_169/add_3AddV2*batch_instance_normalization_169/mul_4:z:0=batch_instance_normalization_169/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?v
LeakyRelu_1	LeakyRelu*batch_instance_normalization_169/add_3:z:0*0
_output_shapes
:?????????  ??
 conv2d_216/Conv2D/ReadVariableOpReadVariableOp)conv2d_216_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_216/Conv2DConv2DLeakyRelu_1:activations:0(conv2d_216/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
?batch_instance_normalization_170/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-batch_instance_normalization_170/moments/meanMeanconv2d_216/Conv2D:output:0Hbatch_instance_normalization_170/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
5batch_instance_normalization_170/moments/StopGradientStopGradient6batch_instance_normalization_170/moments/mean:output:0*
T0*'
_output_shapes
:??
:batch_instance_normalization_170/moments/SquaredDifferenceSquaredDifferenceconv2d_216/Conv2D:output:0>batch_instance_normalization_170/moments/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Cbatch_instance_normalization_170/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1batch_instance_normalization_170/moments/varianceMean>batch_instance_normalization_170/moments/SquaredDifference:z:0Lbatch_instance_normalization_170/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
$batch_instance_normalization_170/subSubconv2d_216/Conv2D:output:06batch_instance_normalization_170/moments/mean:output:0*
T0*0
_output_shapes
:??????????k
&batch_instance_normalization_170/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
$batch_instance_normalization_170/addAddV2:batch_instance_normalization_170/moments/variance:output:0/batch_instance_normalization_170/add/y:output:0*
T0*'
_output_shapes
:??
&batch_instance_normalization_170/RsqrtRsqrt(batch_instance_normalization_170/add:z:0*
T0*'
_output_shapes
:??
$batch_instance_normalization_170/mulMul(batch_instance_normalization_170/sub:z:0*batch_instance_normalization_170/Rsqrt:y:0*
T0*0
_output_shapes
:???????????
Abatch_instance_normalization_170/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
/batch_instance_normalization_170/moments_1/meanMeanconv2d_216/Conv2D:output:0Jbatch_instance_normalization_170/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
7batch_instance_normalization_170/moments_1/StopGradientStopGradient8batch_instance_normalization_170/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
<batch_instance_normalization_170/moments_1/SquaredDifferenceSquaredDifferenceconv2d_216/Conv2D:output:0@batch_instance_normalization_170/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Ebatch_instance_normalization_170/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
3batch_instance_normalization_170/moments_1/varianceMean@batch_instance_normalization_170/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_170/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
&batch_instance_normalization_170/sub_1Subconv2d_216/Conv2D:output:08batch_instance_normalization_170/moments_1/mean:output:0*
T0*0
_output_shapes
:??????????m
(batch_instance_normalization_170/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&batch_instance_normalization_170/add_1AddV2<batch_instance_normalization_170/moments_1/variance:output:01batch_instance_normalization_170/add_1/y:output:0*
T0*0
_output_shapes
:???????????
(batch_instance_normalization_170/Rsqrt_1Rsqrt*batch_instance_normalization_170/add_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_170/mul_1Mul*batch_instance_normalization_170/sub_1:z:0,batch_instance_normalization_170/Rsqrt_1:y:0*
T0*0
_output_shapes
:???????????
/batch_instance_normalization_170/ReadVariableOpReadVariableOp8batch_instance_normalization_170_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_170/mul_2Mul7batch_instance_normalization_170/ReadVariableOp:value:0(batch_instance_normalization_170/mul:z:0*
T0*0
_output_shapes
:???????????
1batch_instance_normalization_170/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_170_readvariableop_resource*
_output_shapes	
:?*
dtype0m
(batch_instance_normalization_170/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&batch_instance_normalization_170/sub_2Sub1batch_instance_normalization_170/sub_2/x:output:09batch_instance_normalization_170/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
&batch_instance_normalization_170/mul_3Mul*batch_instance_normalization_170/sub_2:z:0*batch_instance_normalization_170/mul_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_170/add_2AddV2*batch_instance_normalization_170/mul_2:z:0*batch_instance_normalization_170/mul_3:z:0*
T0*0
_output_shapes
:???????????
5batch_instance_normalization_170/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_170_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_170/mul_4Mul*batch_instance_normalization_170/add_2:z:0=batch_instance_normalization_170/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5batch_instance_normalization_170/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_170_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_170/add_3AddV2*batch_instance_normalization_170/mul_4:z:0=batch_instance_normalization_170/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????v
LeakyRelu_2	LeakyRelu*batch_instance_normalization_170/add_3:z:0*0
_output_shapes
:???????????
zero_padding2d_16/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ?
zero_padding2d_16/PadPadLeakyRelu_2:activations:0'zero_padding2d_16/Pad/paddings:output:0*
T0*0
_output_shapes
:???????????
 conv2d_217/Conv2D/ReadVariableOpReadVariableOp)conv2d_217_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_217/Conv2DConv2Dzero_padding2d_16/Pad:output:0(conv2d_217/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
?batch_instance_normalization_171/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-batch_instance_normalization_171/moments/meanMeanconv2d_217/Conv2D:output:0Hbatch_instance_normalization_171/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
5batch_instance_normalization_171/moments/StopGradientStopGradient6batch_instance_normalization_171/moments/mean:output:0*
T0*'
_output_shapes
:??
:batch_instance_normalization_171/moments/SquaredDifferenceSquaredDifferenceconv2d_217/Conv2D:output:0>batch_instance_normalization_171/moments/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Cbatch_instance_normalization_171/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1batch_instance_normalization_171/moments/varianceMean>batch_instance_normalization_171/moments/SquaredDifference:z:0Lbatch_instance_normalization_171/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
$batch_instance_normalization_171/subSubconv2d_217/Conv2D:output:06batch_instance_normalization_171/moments/mean:output:0*
T0*0
_output_shapes
:??????????k
&batch_instance_normalization_171/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
$batch_instance_normalization_171/addAddV2:batch_instance_normalization_171/moments/variance:output:0/batch_instance_normalization_171/add/y:output:0*
T0*'
_output_shapes
:??
&batch_instance_normalization_171/RsqrtRsqrt(batch_instance_normalization_171/add:z:0*
T0*'
_output_shapes
:??
$batch_instance_normalization_171/mulMul(batch_instance_normalization_171/sub:z:0*batch_instance_normalization_171/Rsqrt:y:0*
T0*0
_output_shapes
:???????????
Abatch_instance_normalization_171/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
/batch_instance_normalization_171/moments_1/meanMeanconv2d_217/Conv2D:output:0Jbatch_instance_normalization_171/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
7batch_instance_normalization_171/moments_1/StopGradientStopGradient8batch_instance_normalization_171/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
<batch_instance_normalization_171/moments_1/SquaredDifferenceSquaredDifferenceconv2d_217/Conv2D:output:0@batch_instance_normalization_171/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Ebatch_instance_normalization_171/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
3batch_instance_normalization_171/moments_1/varianceMean@batch_instance_normalization_171/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_171/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
&batch_instance_normalization_171/sub_1Subconv2d_217/Conv2D:output:08batch_instance_normalization_171/moments_1/mean:output:0*
T0*0
_output_shapes
:??????????m
(batch_instance_normalization_171/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&batch_instance_normalization_171/add_1AddV2<batch_instance_normalization_171/moments_1/variance:output:01batch_instance_normalization_171/add_1/y:output:0*
T0*0
_output_shapes
:???????????
(batch_instance_normalization_171/Rsqrt_1Rsqrt*batch_instance_normalization_171/add_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_171/mul_1Mul*batch_instance_normalization_171/sub_1:z:0,batch_instance_normalization_171/Rsqrt_1:y:0*
T0*0
_output_shapes
:???????????
/batch_instance_normalization_171/ReadVariableOpReadVariableOp8batch_instance_normalization_171_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_171/mul_2Mul7batch_instance_normalization_171/ReadVariableOp:value:0(batch_instance_normalization_171/mul:z:0*
T0*0
_output_shapes
:???????????
1batch_instance_normalization_171/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_171_readvariableop_resource*
_output_shapes	
:?*
dtype0m
(batch_instance_normalization_171/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&batch_instance_normalization_171/sub_2Sub1batch_instance_normalization_171/sub_2/x:output:09batch_instance_normalization_171/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
&batch_instance_normalization_171/mul_3Mul*batch_instance_normalization_171/sub_2:z:0*batch_instance_normalization_171/mul_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_171/add_2AddV2*batch_instance_normalization_171/mul_2:z:0*batch_instance_normalization_171/mul_3:z:0*
T0*0
_output_shapes
:???????????
5batch_instance_normalization_171/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_171_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_171/mul_4Mul*batch_instance_normalization_171/add_2:z:0=batch_instance_normalization_171/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5batch_instance_normalization_171/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_171_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_171/add_3AddV2*batch_instance_normalization_171/mul_4:z:0=batch_instance_normalization_171/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????v
LeakyRelu_3	LeakyRelu*batch_instance_normalization_171/add_3:z:0*0
_output_shapes
:???????????
zero_padding2d_17/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ?
zero_padding2d_17/PadPadLeakyRelu_3:activations:0'zero_padding2d_17/Pad/paddings:output:0*
T0*0
_output_shapes
:???????????
 conv2d_218/Conv2D/ReadVariableOpReadVariableOp)conv2d_218_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_218/Conv2DConv2Dzero_padding2d_17/Pad:output:0(conv2d_218/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
!conv2d_218/BiasAdd/ReadVariableOpReadVariableOp*conv2d_218_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_218/BiasAddBiasAddconv2d_218/Conv2D:output:0)conv2d_218/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????r
IdentityIdentityconv2d_218/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^batch_instance_normalization_169/ReadVariableOp2^batch_instance_normalization_169/ReadVariableOp_16^batch_instance_normalization_169/add_3/ReadVariableOp6^batch_instance_normalization_169/mul_4/ReadVariableOp0^batch_instance_normalization_170/ReadVariableOp2^batch_instance_normalization_170/ReadVariableOp_16^batch_instance_normalization_170/add_3/ReadVariableOp6^batch_instance_normalization_170/mul_4/ReadVariableOp0^batch_instance_normalization_171/ReadVariableOp2^batch_instance_normalization_171/ReadVariableOp_16^batch_instance_normalization_171/add_3/ReadVariableOp6^batch_instance_normalization_171/mul_4/ReadVariableOp!^conv2d_214/Conv2D/ReadVariableOp!^conv2d_215/Conv2D/ReadVariableOp!^conv2d_216/Conv2D/ReadVariableOp!^conv2d_217/Conv2D/ReadVariableOp"^conv2d_218/BiasAdd/ReadVariableOp!^conv2d_218/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 2b
/batch_instance_normalization_169/ReadVariableOp/batch_instance_normalization_169/ReadVariableOp2f
1batch_instance_normalization_169/ReadVariableOp_11batch_instance_normalization_169/ReadVariableOp_12n
5batch_instance_normalization_169/add_3/ReadVariableOp5batch_instance_normalization_169/add_3/ReadVariableOp2n
5batch_instance_normalization_169/mul_4/ReadVariableOp5batch_instance_normalization_169/mul_4/ReadVariableOp2b
/batch_instance_normalization_170/ReadVariableOp/batch_instance_normalization_170/ReadVariableOp2f
1batch_instance_normalization_170/ReadVariableOp_11batch_instance_normalization_170/ReadVariableOp_12n
5batch_instance_normalization_170/add_3/ReadVariableOp5batch_instance_normalization_170/add_3/ReadVariableOp2n
5batch_instance_normalization_170/mul_4/ReadVariableOp5batch_instance_normalization_170/mul_4/ReadVariableOp2b
/batch_instance_normalization_171/ReadVariableOp/batch_instance_normalization_171/ReadVariableOp2f
1batch_instance_normalization_171/ReadVariableOp_11batch_instance_normalization_171/ReadVariableOp_12n
5batch_instance_normalization_171/add_3/ReadVariableOp5batch_instance_normalization_171/add_3/ReadVariableOp2n
5batch_instance_normalization_171/mul_4/ReadVariableOp5batch_instance_normalization_171/mul_4/ReadVariableOp2D
 conv2d_214/Conv2D/ReadVariableOp conv2d_214/Conv2D/ReadVariableOp2D
 conv2d_215/Conv2D/ReadVariableOp conv2d_215/Conv2D/ReadVariableOp2D
 conv2d_216/Conv2D/ReadVariableOp conv2d_216/Conv2D/ReadVariableOp2D
 conv2d_217/Conv2D/ReadVariableOp conv2d_217/Conv2D/ReadVariableOp2F
!conv2d_218/BiasAdd/ReadVariableOp!conv2d_218/BiasAdd/ReadVariableOp2D
 conv2d_218/Conv2D/ReadVariableOp conv2d_218/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?<
?	
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547255
input_1-
conv2d_214_56547210:@.
conv2d_215_56547214:@?8
)batch_instance_normalization_169_56547217:	?8
)batch_instance_normalization_169_56547219:	?8
)batch_instance_normalization_169_56547221:	?/
conv2d_216_56547225:??8
)batch_instance_normalization_170_56547228:	?8
)batch_instance_normalization_170_56547230:	?8
)batch_instance_normalization_170_56547232:	?/
conv2d_217_56547237:??8
)batch_instance_normalization_171_56547240:	?8
)batch_instance_normalization_171_56547242:	?8
)batch_instance_normalization_171_56547244:	?.
conv2d_218_56547249:?!
conv2d_218_56547251:
identity??8batch_instance_normalization_169/StatefulPartitionedCall?8batch_instance_normalization_170/StatefulPartitionedCall?8batch_instance_normalization_171/StatefulPartitionedCall?"conv2d_214/StatefulPartitionedCall?"conv2d_215/StatefulPartitionedCall?"conv2d_216/StatefulPartitionedCall?"conv2d_217/StatefulPartitionedCall?"conv2d_218/StatefulPartitionedCall?
"conv2d_214/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_214_56547210*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56546691t
	LeakyRelu	LeakyRelu+conv2d_214/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@@@?
"conv2d_215/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0conv2d_215_56547214*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56546703?
8batch_instance_normalization_169/StatefulPartitionedCallStatefulPartitionedCall+conv2d_215/StatefulPartitionedCall:output:0)batch_instance_normalization_169_56547217)batch_instance_normalization_169_56547219)batch_instance_normalization_169_56547221*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56546747?
LeakyRelu_1	LeakyReluAbatch_instance_normalization_169/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????  ??
"conv2d_216/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_216_56547225*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56546763?
8batch_instance_normalization_170/StatefulPartitionedCallStatefulPartitionedCall+conv2d_216/StatefulPartitionedCall:output:0)batch_instance_normalization_170_56547228)batch_instance_normalization_170_56547230)batch_instance_normalization_170_56547232*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56546807?
LeakyRelu_2	LeakyReluAbatch_instance_normalization_170/StatefulPartitionedCall:output:0*0
_output_shapes
:???????????
!zero_padding2d_16/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56546661?
"conv2d_217/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_16/PartitionedCall:output:0conv2d_217_56547237*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56546824?
8batch_instance_normalization_171/StatefulPartitionedCallStatefulPartitionedCall+conv2d_217/StatefulPartitionedCall:output:0)batch_instance_normalization_171_56547240)batch_instance_normalization_171_56547242)batch_instance_normalization_171_56547244*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56546868?
LeakyRelu_3	LeakyReluAbatch_instance_normalization_171/StatefulPartitionedCall:output:0*0
_output_shapes
:???????????
!zero_padding2d_17/PartitionedCallPartitionedCallLeakyRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56546674?
"conv2d_218/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_17/PartitionedCall:output:0conv2d_218_56547249conv2d_218_56547251*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56546888?
IdentityIdentity+conv2d_218/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp9^batch_instance_normalization_169/StatefulPartitionedCall9^batch_instance_normalization_170/StatefulPartitionedCall9^batch_instance_normalization_171/StatefulPartitionedCall#^conv2d_214/StatefulPartitionedCall#^conv2d_215/StatefulPartitionedCall#^conv2d_216/StatefulPartitionedCall#^conv2d_217/StatefulPartitionedCall#^conv2d_218/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 2t
8batch_instance_normalization_169/StatefulPartitionedCall8batch_instance_normalization_169/StatefulPartitionedCall2t
8batch_instance_normalization_170/StatefulPartitionedCall8batch_instance_normalization_170/StatefulPartitionedCall2t
8batch_instance_normalization_171/StatefulPartitionedCall8batch_instance_normalization_171/StatefulPartitionedCall2H
"conv2d_214/StatefulPartitionedCall"conv2d_214/StatefulPartitionedCall2H
"conv2d_215/StatefulPartitionedCall"conv2d_215/StatefulPartitionedCall2H
"conv2d_216/StatefulPartitionedCall"conv2d_216/StatefulPartitionedCall2H
"conv2d_217/StatefulPartitionedCall"conv2d_217/StatefulPartitionedCall2H
"conv2d_218/StatefulPartitionedCall"conv2d_218/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?$
?
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56547858
x&
readvariableop_resource:	?,
mul_4_readvariableop_resource:	?,
add_3_readvariableop_resource:	?
identity??ReadVariableOp?ReadVariableOp_1?add_3/ReadVariableOp?mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:??
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:??????????w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:??????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:?I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:?Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:??????????q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:???????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:??????????u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:??????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:??????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:??????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:??????????c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:??????????e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:?]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:??????????_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:??????????o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:?*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:???????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":??????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:??????????

_user_specified_namex
?*
?
!__inference__traced_save_56547956
file_prefix0
,savev2_conv2d_214_kernel_read_readvariableop0
,savev2_conv2d_215_kernel_read_readvariableopC
?savev2_batch_instance_normalization_169_rho_read_readvariableopE
Asavev2_batch_instance_normalization_169_gamma_read_readvariableopD
@savev2_batch_instance_normalization_169_beta_read_readvariableop0
,savev2_conv2d_216_kernel_read_readvariableopC
?savev2_batch_instance_normalization_170_rho_read_readvariableopE
Asavev2_batch_instance_normalization_170_gamma_read_readvariableopD
@savev2_batch_instance_normalization_170_beta_read_readvariableop0
,savev2_conv2d_217_kernel_read_readvariableopC
?savev2_batch_instance_normalization_171_rho_read_readvariableopE
Asavev2_batch_instance_normalization_171_gamma_read_readvariableopD
@savev2_batch_instance_normalization_171_beta_read_readvariableop0
,savev2_conv2d_218_kernel_read_readvariableop.
*savev2_conv2d_218_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv4_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn4_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn4_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn4_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv5_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'conv5_1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_214_kernel_read_readvariableop,savev2_conv2d_215_kernel_read_readvariableop?savev2_batch_instance_normalization_169_rho_read_readvariableopAsavev2_batch_instance_normalization_169_gamma_read_readvariableop@savev2_batch_instance_normalization_169_beta_read_readvariableop,savev2_conv2d_216_kernel_read_readvariableop?savev2_batch_instance_normalization_170_rho_read_readvariableopAsavev2_batch_instance_normalization_170_gamma_read_readvariableop@savev2_batch_instance_normalization_170_beta_read_readvariableop,savev2_conv2d_217_kernel_read_readvariableop?savev2_batch_instance_normalization_171_rho_read_readvariableopAsavev2_batch_instance_normalization_171_gamma_read_readvariableop@savev2_batch_instance_normalization_171_beta_read_readvariableop,savev2_conv2d_218_kernel_read_readvariableop*savev2_conv2d_218_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@?:?:?:?:??:?:?:?:??:?:?:?:?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!	

_output_shapes	
:?:.
*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:-)
'
_output_shapes
:?: 

_output_shapes
::

_output_shapes
: 
?
?
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56546763

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
-__inference_conv2d_215_layer_call_fn_56547659

inputs"
unknown:@?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56546703x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@@: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56547731

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56547888

inputs9
conv2d_readvariableop_resource:?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_217_layer_call_fn_56547800

inputs#
unknown:??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56546824x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547601

inputsC
)conv2d_214_conv2d_readvariableop_resource:@D
)conv2d_215_conv2d_readvariableop_resource:@?G
8batch_instance_normalization_169_readvariableop_resource:	?M
>batch_instance_normalization_169_mul_4_readvariableop_resource:	?M
>batch_instance_normalization_169_add_3_readvariableop_resource:	?E
)conv2d_216_conv2d_readvariableop_resource:??G
8batch_instance_normalization_170_readvariableop_resource:	?M
>batch_instance_normalization_170_mul_4_readvariableop_resource:	?M
>batch_instance_normalization_170_add_3_readvariableop_resource:	?E
)conv2d_217_conv2d_readvariableop_resource:??G
8batch_instance_normalization_171_readvariableop_resource:	?M
>batch_instance_normalization_171_mul_4_readvariableop_resource:	?M
>batch_instance_normalization_171_add_3_readvariableop_resource:	?D
)conv2d_218_conv2d_readvariableop_resource:?8
*conv2d_218_biasadd_readvariableop_resource:
identity??/batch_instance_normalization_169/ReadVariableOp?1batch_instance_normalization_169/ReadVariableOp_1?5batch_instance_normalization_169/add_3/ReadVariableOp?5batch_instance_normalization_169/mul_4/ReadVariableOp?/batch_instance_normalization_170/ReadVariableOp?1batch_instance_normalization_170/ReadVariableOp_1?5batch_instance_normalization_170/add_3/ReadVariableOp?5batch_instance_normalization_170/mul_4/ReadVariableOp?/batch_instance_normalization_171/ReadVariableOp?1batch_instance_normalization_171/ReadVariableOp_1?5batch_instance_normalization_171/add_3/ReadVariableOp?5batch_instance_normalization_171/mul_4/ReadVariableOp? conv2d_214/Conv2D/ReadVariableOp? conv2d_215/Conv2D/ReadVariableOp? conv2d_216/Conv2D/ReadVariableOp? conv2d_217/Conv2D/ReadVariableOp?!conv2d_218/BiasAdd/ReadVariableOp? conv2d_218/Conv2D/ReadVariableOp?
 conv2d_214/Conv2D/ReadVariableOpReadVariableOp)conv2d_214_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_214/Conv2DConv2Dinputs(conv2d_214/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
c
	LeakyRelu	LeakyReluconv2d_214/Conv2D:output:0*/
_output_shapes
:?????????@@@?
 conv2d_215/Conv2D/ReadVariableOpReadVariableOp)conv2d_215_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_215/Conv2DConv2DLeakyRelu:activations:0(conv2d_215/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
?batch_instance_normalization_169/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-batch_instance_normalization_169/moments/meanMeanconv2d_215/Conv2D:output:0Hbatch_instance_normalization_169/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
5batch_instance_normalization_169/moments/StopGradientStopGradient6batch_instance_normalization_169/moments/mean:output:0*
T0*'
_output_shapes
:??
:batch_instance_normalization_169/moments/SquaredDifferenceSquaredDifferenceconv2d_215/Conv2D:output:0>batch_instance_normalization_169/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ??
Cbatch_instance_normalization_169/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1batch_instance_normalization_169/moments/varianceMean>batch_instance_normalization_169/moments/SquaredDifference:z:0Lbatch_instance_normalization_169/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
$batch_instance_normalization_169/subSubconv2d_215/Conv2D:output:06batch_instance_normalization_169/moments/mean:output:0*
T0*0
_output_shapes
:?????????  ?k
&batch_instance_normalization_169/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
$batch_instance_normalization_169/addAddV2:batch_instance_normalization_169/moments/variance:output:0/batch_instance_normalization_169/add/y:output:0*
T0*'
_output_shapes
:??
&batch_instance_normalization_169/RsqrtRsqrt(batch_instance_normalization_169/add:z:0*
T0*'
_output_shapes
:??
$batch_instance_normalization_169/mulMul(batch_instance_normalization_169/sub:z:0*batch_instance_normalization_169/Rsqrt:y:0*
T0*0
_output_shapes
:?????????  ??
Abatch_instance_normalization_169/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
/batch_instance_normalization_169/moments_1/meanMeanconv2d_215/Conv2D:output:0Jbatch_instance_normalization_169/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
7batch_instance_normalization_169/moments_1/StopGradientStopGradient8batch_instance_normalization_169/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
<batch_instance_normalization_169/moments_1/SquaredDifferenceSquaredDifferenceconv2d_215/Conv2D:output:0@batch_instance_normalization_169/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ??
Ebatch_instance_normalization_169/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
3batch_instance_normalization_169/moments_1/varianceMean@batch_instance_normalization_169/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_169/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
&batch_instance_normalization_169/sub_1Subconv2d_215/Conv2D:output:08batch_instance_normalization_169/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????  ?m
(batch_instance_normalization_169/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&batch_instance_normalization_169/add_1AddV2<batch_instance_normalization_169/moments_1/variance:output:01batch_instance_normalization_169/add_1/y:output:0*
T0*0
_output_shapes
:???????????
(batch_instance_normalization_169/Rsqrt_1Rsqrt*batch_instance_normalization_169/add_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_169/mul_1Mul*batch_instance_normalization_169/sub_1:z:0,batch_instance_normalization_169/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  ??
/batch_instance_normalization_169/ReadVariableOpReadVariableOp8batch_instance_normalization_169_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_169/mul_2Mul7batch_instance_normalization_169/ReadVariableOp:value:0(batch_instance_normalization_169/mul:z:0*
T0*0
_output_shapes
:?????????  ??
1batch_instance_normalization_169/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_169_readvariableop_resource*
_output_shapes	
:?*
dtype0m
(batch_instance_normalization_169/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&batch_instance_normalization_169/sub_2Sub1batch_instance_normalization_169/sub_2/x:output:09batch_instance_normalization_169/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
&batch_instance_normalization_169/mul_3Mul*batch_instance_normalization_169/sub_2:z:0*batch_instance_normalization_169/mul_1:z:0*
T0*0
_output_shapes
:?????????  ??
&batch_instance_normalization_169/add_2AddV2*batch_instance_normalization_169/mul_2:z:0*batch_instance_normalization_169/mul_3:z:0*
T0*0
_output_shapes
:?????????  ??
5batch_instance_normalization_169/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_169_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_169/mul_4Mul*batch_instance_normalization_169/add_2:z:0=batch_instance_normalization_169/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
5batch_instance_normalization_169/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_169_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_169/add_3AddV2*batch_instance_normalization_169/mul_4:z:0=batch_instance_normalization_169/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?v
LeakyRelu_1	LeakyRelu*batch_instance_normalization_169/add_3:z:0*0
_output_shapes
:?????????  ??
 conv2d_216/Conv2D/ReadVariableOpReadVariableOp)conv2d_216_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_216/Conv2DConv2DLeakyRelu_1:activations:0(conv2d_216/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
?batch_instance_normalization_170/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-batch_instance_normalization_170/moments/meanMeanconv2d_216/Conv2D:output:0Hbatch_instance_normalization_170/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
5batch_instance_normalization_170/moments/StopGradientStopGradient6batch_instance_normalization_170/moments/mean:output:0*
T0*'
_output_shapes
:??
:batch_instance_normalization_170/moments/SquaredDifferenceSquaredDifferenceconv2d_216/Conv2D:output:0>batch_instance_normalization_170/moments/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Cbatch_instance_normalization_170/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1batch_instance_normalization_170/moments/varianceMean>batch_instance_normalization_170/moments/SquaredDifference:z:0Lbatch_instance_normalization_170/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
$batch_instance_normalization_170/subSubconv2d_216/Conv2D:output:06batch_instance_normalization_170/moments/mean:output:0*
T0*0
_output_shapes
:??????????k
&batch_instance_normalization_170/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
$batch_instance_normalization_170/addAddV2:batch_instance_normalization_170/moments/variance:output:0/batch_instance_normalization_170/add/y:output:0*
T0*'
_output_shapes
:??
&batch_instance_normalization_170/RsqrtRsqrt(batch_instance_normalization_170/add:z:0*
T0*'
_output_shapes
:??
$batch_instance_normalization_170/mulMul(batch_instance_normalization_170/sub:z:0*batch_instance_normalization_170/Rsqrt:y:0*
T0*0
_output_shapes
:???????????
Abatch_instance_normalization_170/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
/batch_instance_normalization_170/moments_1/meanMeanconv2d_216/Conv2D:output:0Jbatch_instance_normalization_170/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
7batch_instance_normalization_170/moments_1/StopGradientStopGradient8batch_instance_normalization_170/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
<batch_instance_normalization_170/moments_1/SquaredDifferenceSquaredDifferenceconv2d_216/Conv2D:output:0@batch_instance_normalization_170/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Ebatch_instance_normalization_170/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
3batch_instance_normalization_170/moments_1/varianceMean@batch_instance_normalization_170/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_170/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
&batch_instance_normalization_170/sub_1Subconv2d_216/Conv2D:output:08batch_instance_normalization_170/moments_1/mean:output:0*
T0*0
_output_shapes
:??????????m
(batch_instance_normalization_170/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&batch_instance_normalization_170/add_1AddV2<batch_instance_normalization_170/moments_1/variance:output:01batch_instance_normalization_170/add_1/y:output:0*
T0*0
_output_shapes
:???????????
(batch_instance_normalization_170/Rsqrt_1Rsqrt*batch_instance_normalization_170/add_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_170/mul_1Mul*batch_instance_normalization_170/sub_1:z:0,batch_instance_normalization_170/Rsqrt_1:y:0*
T0*0
_output_shapes
:???????????
/batch_instance_normalization_170/ReadVariableOpReadVariableOp8batch_instance_normalization_170_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_170/mul_2Mul7batch_instance_normalization_170/ReadVariableOp:value:0(batch_instance_normalization_170/mul:z:0*
T0*0
_output_shapes
:???????????
1batch_instance_normalization_170/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_170_readvariableop_resource*
_output_shapes	
:?*
dtype0m
(batch_instance_normalization_170/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&batch_instance_normalization_170/sub_2Sub1batch_instance_normalization_170/sub_2/x:output:09batch_instance_normalization_170/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
&batch_instance_normalization_170/mul_3Mul*batch_instance_normalization_170/sub_2:z:0*batch_instance_normalization_170/mul_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_170/add_2AddV2*batch_instance_normalization_170/mul_2:z:0*batch_instance_normalization_170/mul_3:z:0*
T0*0
_output_shapes
:???????????
5batch_instance_normalization_170/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_170_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_170/mul_4Mul*batch_instance_normalization_170/add_2:z:0=batch_instance_normalization_170/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5batch_instance_normalization_170/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_170_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_170/add_3AddV2*batch_instance_normalization_170/mul_4:z:0=batch_instance_normalization_170/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????v
LeakyRelu_2	LeakyRelu*batch_instance_normalization_170/add_3:z:0*0
_output_shapes
:???????????
zero_padding2d_16/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ?
zero_padding2d_16/PadPadLeakyRelu_2:activations:0'zero_padding2d_16/Pad/paddings:output:0*
T0*0
_output_shapes
:???????????
 conv2d_217/Conv2D/ReadVariableOpReadVariableOp)conv2d_217_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_217/Conv2DConv2Dzero_padding2d_16/Pad:output:0(conv2d_217/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
?batch_instance_normalization_171/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
-batch_instance_normalization_171/moments/meanMeanconv2d_217/Conv2D:output:0Hbatch_instance_normalization_171/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
5batch_instance_normalization_171/moments/StopGradientStopGradient6batch_instance_normalization_171/moments/mean:output:0*
T0*'
_output_shapes
:??
:batch_instance_normalization_171/moments/SquaredDifferenceSquaredDifferenceconv2d_217/Conv2D:output:0>batch_instance_normalization_171/moments/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Cbatch_instance_normalization_171/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
1batch_instance_normalization_171/moments/varianceMean>batch_instance_normalization_171/moments/SquaredDifference:z:0Lbatch_instance_normalization_171/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(?
$batch_instance_normalization_171/subSubconv2d_217/Conv2D:output:06batch_instance_normalization_171/moments/mean:output:0*
T0*0
_output_shapes
:??????????k
&batch_instance_normalization_171/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
$batch_instance_normalization_171/addAddV2:batch_instance_normalization_171/moments/variance:output:0/batch_instance_normalization_171/add/y:output:0*
T0*'
_output_shapes
:??
&batch_instance_normalization_171/RsqrtRsqrt(batch_instance_normalization_171/add:z:0*
T0*'
_output_shapes
:??
$batch_instance_normalization_171/mulMul(batch_instance_normalization_171/sub:z:0*batch_instance_normalization_171/Rsqrt:y:0*
T0*0
_output_shapes
:???????????
Abatch_instance_normalization_171/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
/batch_instance_normalization_171/moments_1/meanMeanconv2d_217/Conv2D:output:0Jbatch_instance_normalization_171/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
7batch_instance_normalization_171/moments_1/StopGradientStopGradient8batch_instance_normalization_171/moments_1/mean:output:0*
T0*0
_output_shapes
:???????????
<batch_instance_normalization_171/moments_1/SquaredDifferenceSquaredDifferenceconv2d_217/Conv2D:output:0@batch_instance_normalization_171/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:???????????
Ebatch_instance_normalization_171/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
3batch_instance_normalization_171/moments_1/varianceMean@batch_instance_normalization_171/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_171/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(?
&batch_instance_normalization_171/sub_1Subconv2d_217/Conv2D:output:08batch_instance_normalization_171/moments_1/mean:output:0*
T0*0
_output_shapes
:??????????m
(batch_instance_normalization_171/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7?
&batch_instance_normalization_171/add_1AddV2<batch_instance_normalization_171/moments_1/variance:output:01batch_instance_normalization_171/add_1/y:output:0*
T0*0
_output_shapes
:???????????
(batch_instance_normalization_171/Rsqrt_1Rsqrt*batch_instance_normalization_171/add_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_171/mul_1Mul*batch_instance_normalization_171/sub_1:z:0,batch_instance_normalization_171/Rsqrt_1:y:0*
T0*0
_output_shapes
:???????????
/batch_instance_normalization_171/ReadVariableOpReadVariableOp8batch_instance_normalization_171_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_171/mul_2Mul7batch_instance_normalization_171/ReadVariableOp:value:0(batch_instance_normalization_171/mul:z:0*
T0*0
_output_shapes
:???????????
1batch_instance_normalization_171/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_171_readvariableop_resource*
_output_shapes	
:?*
dtype0m
(batch_instance_normalization_171/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
&batch_instance_normalization_171/sub_2Sub1batch_instance_normalization_171/sub_2/x:output:09batch_instance_normalization_171/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:??
&batch_instance_normalization_171/mul_3Mul*batch_instance_normalization_171/sub_2:z:0*batch_instance_normalization_171/mul_1:z:0*
T0*0
_output_shapes
:???????????
&batch_instance_normalization_171/add_2AddV2*batch_instance_normalization_171/mul_2:z:0*batch_instance_normalization_171/mul_3:z:0*
T0*0
_output_shapes
:???????????
5batch_instance_normalization_171/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_171_mul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_171/mul_4Mul*batch_instance_normalization_171/add_2:z:0=batch_instance_normalization_171/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
5batch_instance_normalization_171/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_171_add_3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
&batch_instance_normalization_171/add_3AddV2*batch_instance_normalization_171/mul_4:z:0=batch_instance_normalization_171/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????v
LeakyRelu_3	LeakyRelu*batch_instance_normalization_171/add_3:z:0*0
_output_shapes
:???????????
zero_padding2d_17/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ?
zero_padding2d_17/PadPadLeakyRelu_3:activations:0'zero_padding2d_17/Pad/paddings:output:0*
T0*0
_output_shapes
:???????????
 conv2d_218/Conv2D/ReadVariableOpReadVariableOp)conv2d_218_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype0?
conv2d_218/Conv2DConv2Dzero_padding2d_17/Pad:output:0(conv2d_218/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
?
!conv2d_218/BiasAdd/ReadVariableOpReadVariableOp*conv2d_218_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_218/BiasAddBiasAddconv2d_218/Conv2D:output:0)conv2d_218/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????r
IdentityIdentityconv2d_218/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp0^batch_instance_normalization_169/ReadVariableOp2^batch_instance_normalization_169/ReadVariableOp_16^batch_instance_normalization_169/add_3/ReadVariableOp6^batch_instance_normalization_169/mul_4/ReadVariableOp0^batch_instance_normalization_170/ReadVariableOp2^batch_instance_normalization_170/ReadVariableOp_16^batch_instance_normalization_170/add_3/ReadVariableOp6^batch_instance_normalization_170/mul_4/ReadVariableOp0^batch_instance_normalization_171/ReadVariableOp2^batch_instance_normalization_171/ReadVariableOp_16^batch_instance_normalization_171/add_3/ReadVariableOp6^batch_instance_normalization_171/mul_4/ReadVariableOp!^conv2d_214/Conv2D/ReadVariableOp!^conv2d_215/Conv2D/ReadVariableOp!^conv2d_216/Conv2D/ReadVariableOp!^conv2d_217/Conv2D/ReadVariableOp"^conv2d_218/BiasAdd/ReadVariableOp!^conv2d_218/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 2b
/batch_instance_normalization_169/ReadVariableOp/batch_instance_normalization_169/ReadVariableOp2f
1batch_instance_normalization_169/ReadVariableOp_11batch_instance_normalization_169/ReadVariableOp_12n
5batch_instance_normalization_169/add_3/ReadVariableOp5batch_instance_normalization_169/add_3/ReadVariableOp2n
5batch_instance_normalization_169/mul_4/ReadVariableOp5batch_instance_normalization_169/mul_4/ReadVariableOp2b
/batch_instance_normalization_170/ReadVariableOp/batch_instance_normalization_170/ReadVariableOp2f
1batch_instance_normalization_170/ReadVariableOp_11batch_instance_normalization_170/ReadVariableOp_12n
5batch_instance_normalization_170/add_3/ReadVariableOp5batch_instance_normalization_170/add_3/ReadVariableOp2n
5batch_instance_normalization_170/mul_4/ReadVariableOp5batch_instance_normalization_170/mul_4/ReadVariableOp2b
/batch_instance_normalization_171/ReadVariableOp/batch_instance_normalization_171/ReadVariableOp2f
1batch_instance_normalization_171/ReadVariableOp_11batch_instance_normalization_171/ReadVariableOp_12n
5batch_instance_normalization_171/add_3/ReadVariableOp5batch_instance_normalization_171/add_3/ReadVariableOp2n
5batch_instance_normalization_171/mul_4/ReadVariableOp5batch_instance_normalization_171/mul_4/ReadVariableOp2D
 conv2d_214/Conv2D/ReadVariableOp conv2d_214/Conv2D/ReadVariableOp2D
 conv2d_215/Conv2D/ReadVariableOp conv2d_215/Conv2D/ReadVariableOp2D
 conv2d_216/Conv2D/ReadVariableOp conv2d_216/Conv2D/ReadVariableOp2D
 conv2d_217/Conv2D/ReadVariableOp conv2d_217/Conv2D/ReadVariableOp2F
!conv2d_218/BiasAdd/ReadVariableOp!conv2d_218/BiasAdd/ReadVariableOp2D
 conv2d_218/Conv2D/ReadVariableOp conv2d_218/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?<
?	
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547207
input_1-
conv2d_214_56547162:@.
conv2d_215_56547166:@?8
)batch_instance_normalization_169_56547169:	?8
)batch_instance_normalization_169_56547171:	?8
)batch_instance_normalization_169_56547173:	?/
conv2d_216_56547177:??8
)batch_instance_normalization_170_56547180:	?8
)batch_instance_normalization_170_56547182:	?8
)batch_instance_normalization_170_56547184:	?/
conv2d_217_56547189:??8
)batch_instance_normalization_171_56547192:	?8
)batch_instance_normalization_171_56547194:	?8
)batch_instance_normalization_171_56547196:	?.
conv2d_218_56547201:?!
conv2d_218_56547203:
identity??8batch_instance_normalization_169/StatefulPartitionedCall?8batch_instance_normalization_170/StatefulPartitionedCall?8batch_instance_normalization_171/StatefulPartitionedCall?"conv2d_214/StatefulPartitionedCall?"conv2d_215/StatefulPartitionedCall?"conv2d_216/StatefulPartitionedCall?"conv2d_217/StatefulPartitionedCall?"conv2d_218/StatefulPartitionedCall?
"conv2d_214/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_214_56547162*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56546691t
	LeakyRelu	LeakyRelu+conv2d_214/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@@@?
"conv2d_215/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0conv2d_215_56547166*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56546703?
8batch_instance_normalization_169/StatefulPartitionedCallStatefulPartitionedCall+conv2d_215/StatefulPartitionedCall:output:0)batch_instance_normalization_169_56547169)batch_instance_normalization_169_56547171)batch_instance_normalization_169_56547173*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56546747?
LeakyRelu_1	LeakyReluAbatch_instance_normalization_169/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????  ??
"conv2d_216/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_216_56547177*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56546763?
8batch_instance_normalization_170/StatefulPartitionedCallStatefulPartitionedCall+conv2d_216/StatefulPartitionedCall:output:0)batch_instance_normalization_170_56547180)batch_instance_normalization_170_56547182)batch_instance_normalization_170_56547184*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56546807?
LeakyRelu_2	LeakyReluAbatch_instance_normalization_170/StatefulPartitionedCall:output:0*0
_output_shapes
:???????????
!zero_padding2d_16/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56546661?
"conv2d_217/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_16/PartitionedCall:output:0conv2d_217_56547189*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56546824?
8batch_instance_normalization_171/StatefulPartitionedCallStatefulPartitionedCall+conv2d_217/StatefulPartitionedCall:output:0)batch_instance_normalization_171_56547192)batch_instance_normalization_171_56547194)batch_instance_normalization_171_56547196*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *g
fbR`
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56546868?
LeakyRelu_3	LeakyReluAbatch_instance_normalization_171/StatefulPartitionedCall:output:0*0
_output_shapes
:???????????
!zero_padding2d_17/PartitionedCallPartitionedCallLeakyRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56546674?
"conv2d_218/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_17/PartitionedCall:output:0conv2d_218_56547201conv2d_218_56547203*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56546888?
IdentityIdentity+conv2d_218/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp9^batch_instance_normalization_169/StatefulPartitionedCall9^batch_instance_normalization_170/StatefulPartitionedCall9^batch_instance_normalization_171/StatefulPartitionedCall#^conv2d_214/StatefulPartitionedCall#^conv2d_215/StatefulPartitionedCall#^conv2d_216/StatefulPartitionedCall#^conv2d_217/StatefulPartitionedCall#^conv2d_218/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:???????????: : : : : : : : : : : : : : : 2t
8batch_instance_normalization_169/StatefulPartitionedCall8batch_instance_normalization_169/StatefulPartitionedCall2t
8batch_instance_normalization_170/StatefulPartitionedCall8batch_instance_normalization_170/StatefulPartitionedCall2t
8batch_instance_normalization_171/StatefulPartitionedCall8batch_instance_normalization_171/StatefulPartitionedCall2H
"conv2d_214/StatefulPartitionedCall"conv2d_214/StatefulPartitionedCall2H
"conv2d_215/StatefulPartitionedCall"conv2d_215/StatefulPartitionedCall2H
"conv2d_216/StatefulPartitionedCall"conv2d_216/StatefulPartitionedCall2H
"conv2d_217/StatefulPartitionedCall"conv2d_217/StatefulPartitionedCall2H
"conv2d_218/StatefulPartitionedCall"conv2d_218/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?$
?
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56547717
x&
readvariableop_resource:	?,
mul_4_readvariableop_resource:	?,
add_3_readvariableop_resource:	?
identity??ReadVariableOp?ReadVariableOp_1?add_3/ReadVariableOp?mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:??
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ?w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:?*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:?????????  ?J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:?I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:?Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:?????????  ?q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:???????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ?u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ?
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:??????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????  ?L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *??'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:??????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:??????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  ?c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:?????????  ?e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:?]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:?????????  ?_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????  ?o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:?*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:?*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:?????????  ??
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????  ?: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:?????????  ?

_user_specified_namex
?
k
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56547793

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             ~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????w
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56546703

inputs9
conv2d_readvariableop_resource:@?
identity??Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????@@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
??
?
$__inference__traced_restore_56548011
file_prefix<
"assignvariableop_conv2d_214_kernel:@?
$assignvariableop_1_conv2d_215_kernel:@?F
7assignvariableop_2_batch_instance_normalization_169_rho:	?H
9assignvariableop_3_batch_instance_normalization_169_gamma:	?G
8assignvariableop_4_batch_instance_normalization_169_beta:	?@
$assignvariableop_5_conv2d_216_kernel:??F
7assignvariableop_6_batch_instance_normalization_170_rho:	?H
9assignvariableop_7_batch_instance_normalization_170_gamma:	?G
8assignvariableop_8_batch_instance_normalization_170_beta:	?@
$assignvariableop_9_conv2d_217_kernel:??G
8assignvariableop_10_batch_instance_normalization_171_rho:	?I
:assignvariableop_11_batch_instance_normalization_171_gamma:	?H
9assignvariableop_12_batch_instance_normalization_171_beta:	?@
%assignvariableop_13_conv2d_218_kernel:?1
#assignvariableop_14_conv2d_218_bias:
identity_16??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv4_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn4_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn4_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn4_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv5_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'conv5_1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_214_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_215_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp7assignvariableop_2_batch_instance_normalization_169_rhoIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_instance_normalization_169_gammaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp8assignvariableop_4_batch_instance_normalization_169_betaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv2d_216_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp7assignvariableop_6_batch_instance_normalization_170_rhoIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_instance_normalization_170_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp8assignvariableop_8_batch_instance_normalization_170_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_conv2d_217_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp8assignvariableop_10_batch_instance_normalization_171_rhoIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_instance_normalization_171_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_batch_instance_normalization_171_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp%assignvariableop_13_conv2d_218_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_218_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_16IdentityIdentity_15:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_16Identity_16:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142(
AssignVariableOp_2AssignVariableOp_22(
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
_user_specified_namefile_prefix"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????D
output_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
conv1_1
conv2_1
	bn2_1
conv3_1
	bn3_1
	zero_pad1
conv4_1
	bn4_1
		zero_pad2

conv5_1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
?
!rho
	"gamma
#beta
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
?

*kernel
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
?
1rho
	2gamma
3beta
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
?
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Grho
	Hgamma
Ibeta
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0
1
!2
"3
#4
*5
16
27
38
@9
G10
H11
I12
V13
W14"
trackable_list_wrapper
?
0
1
!2
"3
#4
*5
16
27
38
@9
G10
H11
I12
V13
W14"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
1__inference_face_d_whole_4_layer_call_fn_56546928
1__inference_face_d_whole_4_layer_call_fn_56547290
1__inference_face_d_whole_4_layer_call_fn_56547325
1__inference_face_d_whole_4_layer_call_fn_56547159?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547463
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547601
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547207
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547255?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_56546651input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
cserving_default"
signature_map
+:)@2conv2d_214/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_conv2d_214_layer_call_fn_56547645?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56547652?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,:*@?2conv2d_215/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
* &call_and_return_all_conditional_losses
& "call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_conv2d_215_layer_call_fn_56547659?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56547666?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
3:1?2$batch_instance_normalization_169/rho
5:3?2&batch_instance_normalization_169/gamma
4:2?2%batch_instance_normalization_169/beta
5
!0
"1
#2"
trackable_list_wrapper
5
!0
"1
#2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
?2?
C__inference_batch_instance_normalization_169_layer_call_fn_56547677?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56547717?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
-:+??2conv2d_216/kernel
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_conv2d_216_layer_call_fn_56547724?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56547731?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
3:1?2$batch_instance_normalization_170/rho
5:3?2&batch_instance_normalization_170/gamma
4:2?2%batch_instance_normalization_170/beta
5
10
21
32"
trackable_list_wrapper
5
10
21
32"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
?2?
C__inference_batch_instance_normalization_170_layer_call_fn_56547742?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56547782?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables

~layers
metrics
 ?layer_regularization_losses
?layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_zero_padding2d_16_layer_call_fn_56547787?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56547793?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
-:+??2conv2d_217/kernel
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_conv2d_217_layer_call_fn_56547800?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56547807?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
3:1?2$batch_instance_normalization_171/rho
5:3?2&batch_instance_normalization_171/gamma
4:2?2%batch_instance_normalization_171/beta
5
G0
H1
I2"
trackable_list_wrapper
5
G0
H1
I2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
?2?
C__inference_batch_instance_normalization_171_layer_call_fn_56547818?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56547858?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
?2?
4__inference_zero_padding2d_17_layer_call_fn_56547863?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56547869?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,:*?2conv2d_218/kernel
:2conv2d_218/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
?2?
-__inference_conv2d_218_layer_call_fn_56547878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56547888?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
f
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
9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
&__inference_signature_wrapper_56547638input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper?
#__inference__wrapped_model_56546651?!"#*123@GHIVW:?7
0?-
+?(
input_1???????????
? ";?8
6
output_1*?'
output_1??????????
^__inference_batch_instance_normalization_169_layer_call_and_return_conditional_losses_56547717j!"#3?0
)?&
$?!
x?????????  ?
? ".?+
$?!
0?????????  ?
? ?
C__inference_batch_instance_normalization_169_layer_call_fn_56547677]!"#3?0
)?&
$?!
x?????????  ?
? "!??????????  ??
^__inference_batch_instance_normalization_170_layer_call_and_return_conditional_losses_56547782j1233?0
)?&
$?!
x??????????
? ".?+
$?!
0??????????
? ?
C__inference_batch_instance_normalization_170_layer_call_fn_56547742]1233?0
)?&
$?!
x??????????
? "!????????????
^__inference_batch_instance_normalization_171_layer_call_and_return_conditional_losses_56547858jGHI3?0
)?&
$?!
x??????????
? ".?+
$?!
0??????????
? ?
C__inference_batch_instance_normalization_171_layer_call_fn_56547818]GHI3?0
)?&
$?!
x??????????
? "!????????????
H__inference_conv2d_214_layer_call_and_return_conditional_losses_56547652m9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????@@@
? ?
-__inference_conv2d_214_layer_call_fn_56547645`9?6
/?,
*?'
inputs???????????
? " ??????????@@@?
H__inference_conv2d_215_layer_call_and_return_conditional_losses_56547666l7?4
-?*
(?%
inputs?????????@@@
? ".?+
$?!
0?????????  ?
? ?
-__inference_conv2d_215_layer_call_fn_56547659_7?4
-?*
(?%
inputs?????????@@@
? "!??????????  ??
H__inference_conv2d_216_layer_call_and_return_conditional_losses_56547731m*8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0??????????
? ?
-__inference_conv2d_216_layer_call_fn_56547724`*8?5
.?+
)?&
inputs?????????  ?
? "!????????????
H__inference_conv2d_217_layer_call_and_return_conditional_losses_56547807m@8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_conv2d_217_layer_call_fn_56547800`@8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_conv2d_218_layer_call_and_return_conditional_losses_56547888mVW8?5
.?+
)?&
inputs??????????
? "-?*
#? 
0?????????
? ?
-__inference_conv2d_218_layer_call_fn_56547878`VW8?5
.?+
)?&
inputs??????????
? " ???????????
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547207?!"#*123@GHIVW>?;
4?1
+?(
input_1???????????
p 
? "-?*
#? 
0?????????
? ?
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547255?!"#*123@GHIVW>?;
4?1
+?(
input_1???????????
p
? "-?*
#? 
0?????????
? ?
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547463!"#*123@GHIVW=?:
3?0
*?'
inputs???????????
p 
? "-?*
#? 
0?????????
? ?
L__inference_face_d_whole_4_layer_call_and_return_conditional_losses_56547601!"#*123@GHIVW=?:
3?0
*?'
inputs???????????
p
? "-?*
#? 
0?????????
? ?
1__inference_face_d_whole_4_layer_call_fn_56546928s!"#*123@GHIVW>?;
4?1
+?(
input_1???????????
p 
? " ???????????
1__inference_face_d_whole_4_layer_call_fn_56547159s!"#*123@GHIVW>?;
4?1
+?(
input_1???????????
p
? " ???????????
1__inference_face_d_whole_4_layer_call_fn_56547290r!"#*123@GHIVW=?:
3?0
*?'
inputs???????????
p 
? " ???????????
1__inference_face_d_whole_4_layer_call_fn_56547325r!"#*123@GHIVW=?:
3?0
*?'
inputs???????????
p
? " ???????????
&__inference_signature_wrapper_56547638?!"#*123@GHIVWE?B
? 
;?8
6
input_1+?(
input_1???????????";?8
6
output_1*?'
output_1??????????
O__inference_zero_padding2d_16_layer_call_and_return_conditional_losses_56547793?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
4__inference_zero_padding2d_16_layer_call_fn_56547787?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
O__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_56547869?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
4__inference_zero_padding2d_17_layer_call_fn_56547863?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????