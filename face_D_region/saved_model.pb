ΤΥ
³
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
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
8
Const
output"dtype"
valuetensor"
dtypetype

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
alphafloat%ΝΜL>"
Ttype0:
2

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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
list(type)(0
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

2	
Α
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68―

conv2d_219/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_219/kernel

%conv2d_219/kernel/Read/ReadVariableOpReadVariableOpconv2d_219/kernel*&
_output_shapes
:@*
dtype0

conv2d_220/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_220/kernel

%conv2d_220/kernel/Read/ReadVariableOpReadVariableOpconv2d_220/kernel*'
_output_shapes
:@*
dtype0
‘
$batch_instance_normalization_172/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_172/rho

8batch_instance_normalization_172/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_172/rho*
_output_shapes	
:*
dtype0
₯
&batch_instance_normalization_172/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_172/gamma

:batch_instance_normalization_172/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_172/gamma*
_output_shapes	
:*
dtype0
£
%batch_instance_normalization_172/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_172/beta

9batch_instance_normalization_172/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_172/beta*
_output_shapes	
:*
dtype0

conv2d_221/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_221/kernel

%conv2d_221/kernel/Read/ReadVariableOpReadVariableOpconv2d_221/kernel*(
_output_shapes
:*
dtype0
‘
$batch_instance_normalization_173/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_173/rho

8batch_instance_normalization_173/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_173/rho*
_output_shapes	
:*
dtype0
₯
&batch_instance_normalization_173/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_173/gamma

:batch_instance_normalization_173/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_173/gamma*
_output_shapes	
:*
dtype0
£
%batch_instance_normalization_173/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_173/beta

9batch_instance_normalization_173/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_173/beta*
_output_shapes	
:*
dtype0

conv2d_222/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_222/kernel

%conv2d_222/kernel/Read/ReadVariableOpReadVariableOpconv2d_222/kernel*(
_output_shapes
:*
dtype0
‘
$batch_instance_normalization_174/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_174/rho

8batch_instance_normalization_174/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_174/rho*
_output_shapes	
:*
dtype0
₯
&batch_instance_normalization_174/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_174/gamma

:batch_instance_normalization_174/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_174/gamma*
_output_shapes	
:*
dtype0
£
%batch_instance_normalization_174/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_174/beta

9batch_instance_normalization_174/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_174/beta*
_output_shapes	
:*
dtype0

conv2d_223/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_223/kernel

%conv2d_223/kernel/Read/ReadVariableOpReadVariableOpconv2d_223/kernel*'
_output_shapes
:*
dtype0
v
conv2d_223/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_223/bias
o
#conv2d_223/bias/Read/ReadVariableOpReadVariableOpconv2d_223/bias*
_output_shapes
:*
dtype0

NoOpNoOp
;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ψ:
valueΞ:BΛ: BΔ:
½
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


kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*


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


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

:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses* 


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

P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
¦

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
°
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
VARIABLE_VALUEconv2d_219/kernel)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

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
VARIABLE_VALUEconv2d_220/kernel)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

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
VARIABLE_VALUE$batch_instance_normalization_172/rho$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_172/gamma&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_172/beta%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

!0
"1
#2*

!0
"1
#2*
* 

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
VARIABLE_VALUEconv2d_221/kernel)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

*0*

*0*
* 

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
VARIABLE_VALUE$batch_instance_normalization_173/rho$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_173/gamma&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_173/beta%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

10
21
32*

10
21
32*
* 

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

}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses* 
* 
* 
TN
VARIABLE_VALUEconv2d_222/kernel)conv4_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

@0*

@0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_174/rho$bn4_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_174/gamma&bn4_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_174/beta%bn4_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1
I2*

G0
H1
I2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 
* 
* 
TN
VARIABLE_VALUEconv2d_223/kernel)conv5_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_223/bias'conv5_1/bias/.ATTRIBUTES/VARIABLE_VALUE*

V0
W1*

V0
W1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
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

serving_default_input_1Placeholder*1
_output_shapes
:?????????*
dtype0*&
shape:?????????

serving_default_input_2Placeholder*1
_output_shapes
:?????????*
dtype0*&
shape:?????????

serving_default_input_3Placeholder*1
_output_shapes
:?????????*
dtype0*&
shape:?????????
ΰ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3conv2d_219/kernelconv2d_220/kernel$batch_instance_normalization_172/rho&batch_instance_normalization_172/gamma%batch_instance_normalization_172/betaconv2d_221/kernel$batch_instance_normalization_173/rho&batch_instance_normalization_173/gamma%batch_instance_normalization_173/betaconv2d_222/kernel$batch_instance_normalization_174/rho&batch_instance_normalization_174/gamma%batch_instance_normalization_174/betaconv2d_223/kernelconv2d_223/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_56549289
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ͺ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_219/kernel/Read/ReadVariableOp%conv2d_220/kernel/Read/ReadVariableOp8batch_instance_normalization_172/rho/Read/ReadVariableOp:batch_instance_normalization_172/gamma/Read/ReadVariableOp9batch_instance_normalization_172/beta/Read/ReadVariableOp%conv2d_221/kernel/Read/ReadVariableOp8batch_instance_normalization_173/rho/Read/ReadVariableOp:batch_instance_normalization_173/gamma/Read/ReadVariableOp9batch_instance_normalization_173/beta/Read/ReadVariableOp%conv2d_222/kernel/Read/ReadVariableOp8batch_instance_normalization_174/rho/Read/ReadVariableOp:batch_instance_normalization_174/gamma/Read/ReadVariableOp9batch_instance_normalization_174/beta/Read/ReadVariableOp%conv2d_223/kernel/Read/ReadVariableOp#conv2d_223/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8 **
f%R#
!__inference__traced_save_56549609
ω
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_219/kernelconv2d_220/kernel$batch_instance_normalization_172/rho&batch_instance_normalization_172/gamma%batch_instance_normalization_172/betaconv2d_221/kernel$batch_instance_normalization_173/rho&batch_instance_normalization_173/gamma%batch_instance_normalization_173/betaconv2d_222/kernel$batch_instance_normalization_174/rho&batch_instance_normalization_174/gamma%batch_instance_normalization_174/betaconv2d_223/kernelconv2d_223/bias*
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
GPU2*0J 8 *-
f(R&
$__inference__traced_restore_56549664Ξ
°


H__inference_conv2d_223_layer_call_and_return_conditional_losses_56549539

inputs9
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
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
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ψ
£
-__inference_conv2d_223_layer_call_fn_56549529

inputs"
unknown:
	unknown_0:
identity’StatefulPartitionedCallθ
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_223_layer_call_and_return_conditional_losses_56548482w
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
 :?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
Γ
P
4__inference_zero_padding2d_18_layer_call_fn_56549438

inputs
identityΰ
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
GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56548244
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
?
Ν
2__inference_face_d_region_4_layer_call_fn_56548919
inputs_0
inputs_1
inputs_2!
unknown:@$
	unknown_0:@
	unknown_1:	
	unknown_2:	
	unknown_3:	%
	unknown_4:
	unknown_5:	
	unknown_6:	
	unknown_7:	%
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	%

unknown_12:

unknown_13:
identity’StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548489w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/2
E


M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548825
input_1
input_2
input_3-
conv2d_219_56548780:@.
conv2d_220_56548784:@8
)batch_instance_normalization_172_56548787:	8
)batch_instance_normalization_172_56548789:	8
)batch_instance_normalization_172_56548791:	/
conv2d_221_56548795:8
)batch_instance_normalization_173_56548798:	8
)batch_instance_normalization_173_56548800:	8
)batch_instance_normalization_173_56548802:	/
conv2d_222_56548807:8
)batch_instance_normalization_174_56548810:	8
)batch_instance_normalization_174_56548812:	8
)batch_instance_normalization_174_56548814:	.
conv2d_223_56548819:!
conv2d_223_56548821:
identity’8batch_instance_normalization_172/StatefulPartitionedCall’8batch_instance_normalization_173/StatefulPartitionedCall’8batch_instance_normalization_174/StatefulPartitionedCall’"conv2d_219/StatefulPartitionedCall’"conv2d_220/StatefulPartitionedCall’"conv2d_221/StatefulPartitionedCall’"conv2d_222/StatefulPartitionedCall’"conv2d_223/StatefulPartitionedCallU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cy
lambda/truedivRealDivinput_2lambda/truediv/y:output:0*
T0*1
_output_shapes
:?????????Q
lambda/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x

lambda/subSublambda/sub/x:output:0lambda/truediv:z:0*
T0*1
_output_shapes
:?????????o
lambda/multiply/mulMulinput_3lambda/sub:z:0*
T0*1
_output_shapes
:?????????u
lambda/multiply_1/mulMulinput_1lambda/truediv:z:0*
T0*1
_output_shapes
:?????????
lambda/add/addAddV2lambda/multiply/mul:z:0lambda/multiply_1/mul:z:0*
T0*1
_output_shapes
:?????????ώ
"conv2d_219/StatefulPartitionedCallStatefulPartitionedCalllambda/add/add:z:0conv2d_219_56548780*
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56548285t
	LeakyRelu	LeakyRelu+conv2d_219/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@@@
"conv2d_220/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0conv2d_220_56548784*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56548297΄
8batch_instance_normalization_172/StatefulPartitionedCallStatefulPartitionedCall+conv2d_220/StatefulPartitionedCall:output:0)batch_instance_normalization_172_56548787)batch_instance_normalization_172_56548789)batch_instance_normalization_172_56548791*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56548341
LeakyRelu_1	LeakyReluAbatch_instance_normalization_172/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????  
"conv2d_221/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_221_56548795*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56548357΄
8batch_instance_normalization_173/StatefulPartitionedCallStatefulPartitionedCall+conv2d_221/StatefulPartitionedCall:output:0)batch_instance_normalization_173_56548798)batch_instance_normalization_173_56548800)batch_instance_normalization_173_56548802*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56548401
LeakyRelu_2	LeakyReluAbatch_instance_normalization_173/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????λ
!zero_padding2d_18/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56548244
"conv2d_222/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_18/PartitionedCall:output:0conv2d_222_56548807*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56548418΄
8batch_instance_normalization_174/StatefulPartitionedCallStatefulPartitionedCall+conv2d_222/StatefulPartitionedCall:output:0)batch_instance_normalization_174_56548810)batch_instance_normalization_174_56548812)batch_instance_normalization_174_56548814*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56548462
LeakyRelu_3	LeakyReluAbatch_instance_normalization_174/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????λ
!zero_padding2d_19/PartitionedCallPartitionedCallLeakyRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56548257­
"conv2d_223/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_19/PartitionedCall:output:0conv2d_223_56548819conv2d_223_56548821*
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_223_layer_call_and_return_conditional_losses_56548482
IdentityIdentity+conv2d_223/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????°
NoOpNoOp9^batch_instance_normalization_172/StatefulPartitionedCall9^batch_instance_normalization_173/StatefulPartitionedCall9^batch_instance_normalization_174/StatefulPartitionedCall#^conv2d_219/StatefulPartitionedCall#^conv2d_220/StatefulPartitionedCall#^conv2d_221/StatefulPartitionedCall#^conv2d_222/StatefulPartitionedCall#^conv2d_223/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 2t
8batch_instance_normalization_172/StatefulPartitionedCall8batch_instance_normalization_172/StatefulPartitionedCall2t
8batch_instance_normalization_173/StatefulPartitionedCall8batch_instance_normalization_173/StatefulPartitionedCall2t
8batch_instance_normalization_174/StatefulPartitionedCall8batch_instance_normalization_174/StatefulPartitionedCall2H
"conv2d_219/StatefulPartitionedCall"conv2d_219/StatefulPartitionedCall2H
"conv2d_220/StatefulPartitionedCall"conv2d_220/StatefulPartitionedCall2H
"conv2d_221/StatefulPartitionedCall"conv2d_221/StatefulPartitionedCall2H
"conv2d_222/StatefulPartitionedCall"conv2d_222/StatefulPartitionedCall2H
"conv2d_223/StatefulPartitionedCall"conv2d_223/StatefulPartitionedCall:Z V
1
_output_shapes
:?????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_2:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_3
°


H__inference_conv2d_223_layer_call_and_return_conditional_losses_56548482

inputs9
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:*
dtype0
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
 :?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
ͺ
»
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56548357

inputs:
conv2d_readvariableop_resource:
identity’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
¦
Ή
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56548285

inputs8
conv2d_readvariableop_resource:@
identity’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
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
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
?
Ν
2__inference_face_d_region_4_layer_call_fn_56548956
inputs_0
inputs_1
inputs_2!
unknown:@$
	unknown_0:@
	unknown_1:	
	unknown_2:	
	unknown_3:	%
	unknown_4:
	unknown_5:	
	unknown_6:	
	unknown_7:	%
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	%

unknown_12:

unknown_13:
identity’StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548698w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/2
Σ

-__inference_conv2d_221_layer_call_fn_56549375

inputs#
unknown:
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56548357x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
Γ
P
4__inference_zero_padding2d_19_layer_call_fn_56549514

inputs
identityΰ
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
GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56548257
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
ν
k
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56548244

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
E


M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548698

inputs
inputs_1
inputs_2-
conv2d_219_56548653:@.
conv2d_220_56548657:@8
)batch_instance_normalization_172_56548660:	8
)batch_instance_normalization_172_56548662:	8
)batch_instance_normalization_172_56548664:	/
conv2d_221_56548668:8
)batch_instance_normalization_173_56548671:	8
)batch_instance_normalization_173_56548673:	8
)batch_instance_normalization_173_56548675:	/
conv2d_222_56548680:8
)batch_instance_normalization_174_56548683:	8
)batch_instance_normalization_174_56548685:	8
)batch_instance_normalization_174_56548687:	.
conv2d_223_56548692:!
conv2d_223_56548694:
identity’8batch_instance_normalization_172/StatefulPartitionedCall’8batch_instance_normalization_173/StatefulPartitionedCall’8batch_instance_normalization_174/StatefulPartitionedCall’"conv2d_219/StatefulPartitionedCall’"conv2d_220/StatefulPartitionedCall’"conv2d_221/StatefulPartitionedCall’"conv2d_222/StatefulPartitionedCall’"conv2d_223/StatefulPartitionedCallU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cz
lambda/truedivRealDivinputs_1lambda/truediv/y:output:0*
T0*1
_output_shapes
:?????????Q
lambda/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x

lambda/subSublambda/sub/x:output:0lambda/truediv:z:0*
T0*1
_output_shapes
:?????????p
lambda/multiply/mulMulinputs_2lambda/sub:z:0*
T0*1
_output_shapes
:?????????t
lambda/multiply_1/mulMulinputslambda/truediv:z:0*
T0*1
_output_shapes
:?????????
lambda/add/addAddV2lambda/multiply/mul:z:0lambda/multiply_1/mul:z:0*
T0*1
_output_shapes
:?????????ώ
"conv2d_219/StatefulPartitionedCallStatefulPartitionedCalllambda/add/add:z:0conv2d_219_56548653*
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56548285t
	LeakyRelu	LeakyRelu+conv2d_219/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@@@
"conv2d_220/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0conv2d_220_56548657*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56548297΄
8batch_instance_normalization_172/StatefulPartitionedCallStatefulPartitionedCall+conv2d_220/StatefulPartitionedCall:output:0)batch_instance_normalization_172_56548660)batch_instance_normalization_172_56548662)batch_instance_normalization_172_56548664*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56548341
LeakyRelu_1	LeakyReluAbatch_instance_normalization_172/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????  
"conv2d_221/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_221_56548668*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56548357΄
8batch_instance_normalization_173/StatefulPartitionedCallStatefulPartitionedCall+conv2d_221/StatefulPartitionedCall:output:0)batch_instance_normalization_173_56548671)batch_instance_normalization_173_56548673)batch_instance_normalization_173_56548675*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56548401
LeakyRelu_2	LeakyReluAbatch_instance_normalization_173/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????λ
!zero_padding2d_18/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56548244
"conv2d_222/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_18/PartitionedCall:output:0conv2d_222_56548680*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56548418΄
8batch_instance_normalization_174/StatefulPartitionedCallStatefulPartitionedCall+conv2d_222/StatefulPartitionedCall:output:0)batch_instance_normalization_174_56548683)batch_instance_normalization_174_56548685)batch_instance_normalization_174_56548687*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56548462
LeakyRelu_3	LeakyReluAbatch_instance_normalization_174/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????λ
!zero_padding2d_19/PartitionedCallPartitionedCallLeakyRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56548257­
"conv2d_223/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_19/PartitionedCall:output:0conv2d_223_56548692conv2d_223_56548694*
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_223_layer_call_and_return_conditional_losses_56548482
IdentityIdentity+conv2d_223/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????°
NoOpNoOp9^batch_instance_normalization_172/StatefulPartitionedCall9^batch_instance_normalization_173/StatefulPartitionedCall9^batch_instance_normalization_174/StatefulPartitionedCall#^conv2d_219/StatefulPartitionedCall#^conv2d_220/StatefulPartitionedCall#^conv2d_221/StatefulPartitionedCall#^conv2d_222/StatefulPartitionedCall#^conv2d_223/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 2t
8batch_instance_normalization_172/StatefulPartitionedCall8batch_instance_normalization_172/StatefulPartitionedCall2t
8batch_instance_normalization_173/StatefulPartitionedCall8batch_instance_normalization_173/StatefulPartitionedCall2t
8batch_instance_normalization_174/StatefulPartitionedCall8batch_instance_normalization_174/StatefulPartitionedCall2H
"conv2d_219/StatefulPartitionedCall"conv2d_219/StatefulPartitionedCall2H
"conv2d_220/StatefulPartitionedCall"conv2d_220/StatefulPartitionedCall2H
"conv2d_221/StatefulPartitionedCall"conv2d_221/StatefulPartitionedCall2H
"conv2d_222/StatefulPartitionedCall"conv2d_222/StatefulPartitionedCall2H
"conv2d_223/StatefulPartitionedCall"conv2d_223/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:?????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:?????????
 
_user_specified_nameinputs
Ρ

-__inference_conv2d_219_layer_call_fn_56549296

inputs!
unknown:@
identity’StatefulPartitionedCallΫ
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56548285w
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
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
ν
k
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56549444

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
E


M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548882
input_1
input_2
input_3-
conv2d_219_56548837:@.
conv2d_220_56548841:@8
)batch_instance_normalization_172_56548844:	8
)batch_instance_normalization_172_56548846:	8
)batch_instance_normalization_172_56548848:	/
conv2d_221_56548852:8
)batch_instance_normalization_173_56548855:	8
)batch_instance_normalization_173_56548857:	8
)batch_instance_normalization_173_56548859:	/
conv2d_222_56548864:8
)batch_instance_normalization_174_56548867:	8
)batch_instance_normalization_174_56548869:	8
)batch_instance_normalization_174_56548871:	.
conv2d_223_56548876:!
conv2d_223_56548878:
identity’8batch_instance_normalization_172/StatefulPartitionedCall’8batch_instance_normalization_173/StatefulPartitionedCall’8batch_instance_normalization_174/StatefulPartitionedCall’"conv2d_219/StatefulPartitionedCall’"conv2d_220/StatefulPartitionedCall’"conv2d_221/StatefulPartitionedCall’"conv2d_222/StatefulPartitionedCall’"conv2d_223/StatefulPartitionedCallU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cy
lambda/truedivRealDivinput_2lambda/truediv/y:output:0*
T0*1
_output_shapes
:?????????Q
lambda/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x

lambda/subSublambda/sub/x:output:0lambda/truediv:z:0*
T0*1
_output_shapes
:?????????o
lambda/multiply/mulMulinput_3lambda/sub:z:0*
T0*1
_output_shapes
:?????????u
lambda/multiply_1/mulMulinput_1lambda/truediv:z:0*
T0*1
_output_shapes
:?????????
lambda/add/addAddV2lambda/multiply/mul:z:0lambda/multiply_1/mul:z:0*
T0*1
_output_shapes
:?????????ώ
"conv2d_219/StatefulPartitionedCallStatefulPartitionedCalllambda/add/add:z:0conv2d_219_56548837*
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56548285t
	LeakyRelu	LeakyRelu+conv2d_219/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@@@
"conv2d_220/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0conv2d_220_56548841*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56548297΄
8batch_instance_normalization_172/StatefulPartitionedCallStatefulPartitionedCall+conv2d_220/StatefulPartitionedCall:output:0)batch_instance_normalization_172_56548844)batch_instance_normalization_172_56548846)batch_instance_normalization_172_56548848*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56548341
LeakyRelu_1	LeakyReluAbatch_instance_normalization_172/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????  
"conv2d_221/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_221_56548852*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56548357΄
8batch_instance_normalization_173/StatefulPartitionedCallStatefulPartitionedCall+conv2d_221/StatefulPartitionedCall:output:0)batch_instance_normalization_173_56548855)batch_instance_normalization_173_56548857)batch_instance_normalization_173_56548859*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56548401
LeakyRelu_2	LeakyReluAbatch_instance_normalization_173/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????λ
!zero_padding2d_18/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56548244
"conv2d_222/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_18/PartitionedCall:output:0conv2d_222_56548864*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56548418΄
8batch_instance_normalization_174/StatefulPartitionedCallStatefulPartitionedCall+conv2d_222/StatefulPartitionedCall:output:0)batch_instance_normalization_174_56548867)batch_instance_normalization_174_56548869)batch_instance_normalization_174_56548871*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56548462
LeakyRelu_3	LeakyReluAbatch_instance_normalization_174/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????λ
!zero_padding2d_19/PartitionedCallPartitionedCallLeakyRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56548257­
"conv2d_223/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_19/PartitionedCall:output:0conv2d_223_56548876conv2d_223_56548878*
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_223_layer_call_and_return_conditional_losses_56548482
IdentityIdentity+conv2d_223/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????°
NoOpNoOp9^batch_instance_normalization_172/StatefulPartitionedCall9^batch_instance_normalization_173/StatefulPartitionedCall9^batch_instance_normalization_174/StatefulPartitionedCall#^conv2d_219/StatefulPartitionedCall#^conv2d_220/StatefulPartitionedCall#^conv2d_221/StatefulPartitionedCall#^conv2d_222/StatefulPartitionedCall#^conv2d_223/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 2t
8batch_instance_normalization_172/StatefulPartitionedCall8batch_instance_normalization_172/StatefulPartitionedCall2t
8batch_instance_normalization_173/StatefulPartitionedCall8batch_instance_normalization_173/StatefulPartitionedCall2t
8batch_instance_normalization_174/StatefulPartitionedCall8batch_instance_normalization_174/StatefulPartitionedCall2H
"conv2d_219/StatefulPartitionedCall"conv2d_219/StatefulPartitionedCall2H
"conv2d_220/StatefulPartitionedCall"conv2d_220/StatefulPartitionedCall2H
"conv2d_221/StatefulPartitionedCall"conv2d_221/StatefulPartitionedCall2H
"conv2d_222/StatefulPartitionedCall"conv2d_222/StatefulPartitionedCall2H
"conv2d_223/StatefulPartitionedCall"conv2d_223/StatefulPartitionedCall:Z V
1
_output_shapes
:?????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_2:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_3
«
»
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56548418

inputs:
conv2d_readvariableop_resource:
identity’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
φ
Κ
2__inference_face_d_region_4_layer_call_fn_56548768
input_1
input_2
input_3!
unknown:@$
	unknown_0:@
	unknown_1:	
	unknown_2:	
	unknown_3:	%
	unknown_4:
	unknown_5:	
	unknown_6:	
	unknown_7:	%
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	%

unknown_12:

unknown_13:
identity’StatefulPartitionedCall―
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548698w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:?????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_2:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_3
Σ

-__inference_conv2d_222_layer_call_fn_56549451

inputs#
unknown:
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56548418x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
·$
Ξ
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56549509
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identity’ReadVariableOp’ReadVariableOp_1’add_3/ReadVariableOp’mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:?????????q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ά
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:?????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:?????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:?????????e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:?????????_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:?????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:?????????

_user_specified_namex
·$
Ξ
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56548341
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identity’ReadVariableOp’ReadVariableOp_1’add_3/ReadVariableOp’mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:?????????  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:?????????  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ά
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:?????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:?????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:?????????  e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:?????????  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:?????????  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:?????????  

_user_specified_namex
ν
k
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56549520

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
·$
Ξ
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56548462
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identity’ReadVariableOp’ReadVariableOp_1’add_3/ReadVariableOp’mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:?????????q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ά
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:?????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:?????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:?????????e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:?????????_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:?????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:?????????

_user_specified_namex
¦
Ή
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56549303

inputs8
conv2d_readvariableop_resource:@
identity’Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
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
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs
ΐ
Ύ
&__inference_signature_wrapper_56549289
input_1
input_2
input_3!
unknown:@$
	unknown_0:@
	unknown_1:	
	unknown_2:	
	unknown_3:	%
	unknown_4:
	unknown_5:	
	unknown_6:	
	unknown_7:	%
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	%

unknown_12:

unknown_13:
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_56548234w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:?????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_2:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_3
ν
k
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56548257

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
΅
Γ
C__inference_batch_instance_normalization_172_layer_call_fn_56549328
x
unknown:	
	unknown_0:	
	unknown_1:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56548341x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????  

_user_specified_namex
·$
Ξ
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56549433
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identity’ReadVariableOp’ReadVariableOp_1’add_3/ReadVariableOp’mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:?????????q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ά
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:?????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:?????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:?????????e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:?????????_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:?????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:?????????

_user_specified_namex
*

!__inference__traced_save_56549609
file_prefix0
,savev2_conv2d_219_kernel_read_readvariableop0
,savev2_conv2d_220_kernel_read_readvariableopC
?savev2_batch_instance_normalization_172_rho_read_readvariableopE
Asavev2_batch_instance_normalization_172_gamma_read_readvariableopD
@savev2_batch_instance_normalization_172_beta_read_readvariableop0
,savev2_conv2d_221_kernel_read_readvariableopC
?savev2_batch_instance_normalization_173_rho_read_readvariableopE
Asavev2_batch_instance_normalization_173_gamma_read_readvariableopD
@savev2_batch_instance_normalization_173_beta_read_readvariableop0
,savev2_conv2d_222_kernel_read_readvariableopC
?savev2_batch_instance_normalization_174_rho_read_readvariableopE
Asavev2_batch_instance_normalization_174_gamma_read_readvariableopD
@savev2_batch_instance_normalization_174_beta_read_readvariableop0
,savev2_conv2d_223_kernel_read_readvariableop.
*savev2_conv2d_223_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
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
: ι
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv4_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn4_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn4_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn4_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv5_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'conv5_1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B £
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_219_kernel_read_readvariableop,savev2_conv2d_220_kernel_read_readvariableop?savev2_batch_instance_normalization_172_rho_read_readvariableopAsavev2_batch_instance_normalization_172_gamma_read_readvariableop@savev2_batch_instance_normalization_172_beta_read_readvariableop,savev2_conv2d_221_kernel_read_readvariableop?savev2_batch_instance_normalization_173_rho_read_readvariableopAsavev2_batch_instance_normalization_173_gamma_read_readvariableop@savev2_batch_instance_normalization_173_beta_read_readvariableop,savev2_conv2d_222_kernel_read_readvariableop?savev2_batch_instance_normalization_174_rho_read_readvariableopAsavev2_batch_instance_normalization_174_gamma_read_readvariableop@savev2_batch_instance_normalization_174_beta_read_readvariableop,savev2_conv2d_223_kernel_read_readvariableop*savev2_conv2d_223_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*Ύ
_input_shapes¬
©: :@:@:::::::::::::: 2(
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
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!	

_output_shapes	
::.
*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::-)
'
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
ήΩ
­
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56549103
inputs_0
inputs_1
inputs_2C
)conv2d_219_conv2d_readvariableop_resource:@D
)conv2d_220_conv2d_readvariableop_resource:@G
8batch_instance_normalization_172_readvariableop_resource:	M
>batch_instance_normalization_172_mul_4_readvariableop_resource:	M
>batch_instance_normalization_172_add_3_readvariableop_resource:	E
)conv2d_221_conv2d_readvariableop_resource:G
8batch_instance_normalization_173_readvariableop_resource:	M
>batch_instance_normalization_173_mul_4_readvariableop_resource:	M
>batch_instance_normalization_173_add_3_readvariableop_resource:	E
)conv2d_222_conv2d_readvariableop_resource:G
8batch_instance_normalization_174_readvariableop_resource:	M
>batch_instance_normalization_174_mul_4_readvariableop_resource:	M
>batch_instance_normalization_174_add_3_readvariableop_resource:	D
)conv2d_223_conv2d_readvariableop_resource:8
*conv2d_223_biasadd_readvariableop_resource:
identity’/batch_instance_normalization_172/ReadVariableOp’1batch_instance_normalization_172/ReadVariableOp_1’5batch_instance_normalization_172/add_3/ReadVariableOp’5batch_instance_normalization_172/mul_4/ReadVariableOp’/batch_instance_normalization_173/ReadVariableOp’1batch_instance_normalization_173/ReadVariableOp_1’5batch_instance_normalization_173/add_3/ReadVariableOp’5batch_instance_normalization_173/mul_4/ReadVariableOp’/batch_instance_normalization_174/ReadVariableOp’1batch_instance_normalization_174/ReadVariableOp_1’5batch_instance_normalization_174/add_3/ReadVariableOp’5batch_instance_normalization_174/mul_4/ReadVariableOp’ conv2d_219/Conv2D/ReadVariableOp’ conv2d_220/Conv2D/ReadVariableOp’ conv2d_221/Conv2D/ReadVariableOp’ conv2d_222/Conv2D/ReadVariableOp’!conv2d_223/BiasAdd/ReadVariableOp’ conv2d_223/Conv2D/ReadVariableOpU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cz
lambda/truedivRealDivinputs_1lambda/truediv/y:output:0*
T0*1
_output_shapes
:?????????Q
lambda/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x

lambda/subSublambda/sub/x:output:0lambda/truediv:z:0*
T0*1
_output_shapes
:?????????p
lambda/multiply/mulMulinputs_2lambda/sub:z:0*
T0*1
_output_shapes
:?????????v
lambda/multiply_1/mulMulinputs_0lambda/truediv:z:0*
T0*1
_output_shapes
:?????????
lambda/add/addAddV2lambda/multiply/mul:z:0lambda/multiply_1/mul:z:0*
T0*1
_output_shapes
:?????????
 conv2d_219/Conv2D/ReadVariableOpReadVariableOp)conv2d_219_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0»
conv2d_219/Conv2DConv2Dlambda/add/add:z:0(conv2d_219/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
c
	LeakyRelu	LeakyReluconv2d_219/Conv2D:output:0*/
_output_shapes
:?????????@@@
 conv2d_220/Conv2D/ReadVariableOpReadVariableOp)conv2d_220_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Α
conv2d_220/Conv2DConv2DLeakyRelu:activations:0(conv2d_220/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides

?batch_instance_normalization_172/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ή
-batch_instance_normalization_172/moments/meanMeanconv2d_220/Conv2D:output:0Hbatch_instance_normalization_172/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(―
5batch_instance_normalization_172/moments/StopGradientStopGradient6batch_instance_normalization_172/moments/mean:output:0*
T0*'
_output_shapes
:ζ
:batch_instance_normalization_172/moments/SquaredDifferenceSquaredDifferenceconv2d_220/Conv2D:output:0>batch_instance_normalization_172/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  
Cbatch_instance_normalization_172/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_172/moments/varianceMean>batch_instance_normalization_172/moments/SquaredDifference:z:0Lbatch_instance_normalization_172/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ί
$batch_instance_normalization_172/subSubconv2d_220/Conv2D:output:06batch_instance_normalization_172/moments/mean:output:0*
T0*0
_output_shapes
:?????????  k
&batch_instance_normalization_172/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Μ
$batch_instance_normalization_172/addAddV2:batch_instance_normalization_172/moments/variance:output:0/batch_instance_normalization_172/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_172/RsqrtRsqrt(batch_instance_normalization_172/add:z:0*
T0*'
_output_shapes
:Ό
$batch_instance_normalization_172/mulMul(batch_instance_normalization_172/sub:z:0*batch_instance_normalization_172/Rsqrt:y:0*
T0*0
_output_shapes
:?????????  
Abatch_instance_normalization_172/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      λ
/batch_instance_normalization_172/moments_1/meanMeanconv2d_220/Conv2D:output:0Jbatch_instance_normalization_172/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ό
7batch_instance_normalization_172/moments_1/StopGradientStopGradient8batch_instance_normalization_172/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????κ
<batch_instance_normalization_172/moments_1/SquaredDifferenceSquaredDifferenceconv2d_220/Conv2D:output:0@batch_instance_normalization_172/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  
Ebatch_instance_normalization_172/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_172/moments_1/varianceMean@batch_instance_normalization_172/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_172/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ύ
&batch_instance_normalization_172/sub_1Subconv2d_220/Conv2D:output:08batch_instance_normalization_172/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????  m
(batch_instance_normalization_172/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Ϋ
&batch_instance_normalization_172/add_1AddV2<batch_instance_normalization_172/moments_1/variance:output:01batch_instance_normalization_172/add_1/y:output:0*
T0*0
_output_shapes
:?????????
(batch_instance_normalization_172/Rsqrt_1Rsqrt*batch_instance_normalization_172/add_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_172/mul_1Mul*batch_instance_normalization_172/sub_1:z:0,batch_instance_normalization_172/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  ₯
/batch_instance_normalization_172/ReadVariableOpReadVariableOp8batch_instance_normalization_172_readvariableop_resource*
_output_shapes	
:*
dtype0Λ
&batch_instance_normalization_172/mul_2Mul7batch_instance_normalization_172/ReadVariableOp:value:0(batch_instance_normalization_172/mul:z:0*
T0*0
_output_shapes
:?????????  §
1batch_instance_normalization_172/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_172_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_172/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Α
&batch_instance_normalization_172/sub_2Sub1batch_instance_normalization_172/sub_2/x:output:09batch_instance_normalization_172/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:ΐ
&batch_instance_normalization_172/mul_3Mul*batch_instance_normalization_172/sub_2:z:0*batch_instance_normalization_172/mul_1:z:0*
T0*0
_output_shapes
:?????????  Β
&batch_instance_normalization_172/add_2AddV2*batch_instance_normalization_172/mul_2:z:0*batch_instance_normalization_172/mul_3:z:0*
T0*0
_output_shapes
:?????????  ±
5batch_instance_normalization_172/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_172_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Σ
&batch_instance_normalization_172/mul_4Mul*batch_instance_normalization_172/add_2:z:0=batch_instance_normalization_172/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ±
5batch_instance_normalization_172/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_172_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Υ
&batch_instance_normalization_172/add_3AddV2*batch_instance_normalization_172/mul_4:z:0=batch_instance_normalization_172/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  v
LeakyRelu_1	LeakyRelu*batch_instance_normalization_172/add_3:z:0*0
_output_shapes
:?????????  
 conv2d_221/Conv2D/ReadVariableOpReadVariableOp)conv2d_221_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Γ
conv2d_221/Conv2DConv2DLeakyRelu_1:activations:0(conv2d_221/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

?batch_instance_normalization_173/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ή
-batch_instance_normalization_173/moments/meanMeanconv2d_221/Conv2D:output:0Hbatch_instance_normalization_173/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(―
5batch_instance_normalization_173/moments/StopGradientStopGradient6batch_instance_normalization_173/moments/mean:output:0*
T0*'
_output_shapes
:ζ
:batch_instance_normalization_173/moments/SquaredDifferenceSquaredDifferenceconv2d_221/Conv2D:output:0>batch_instance_normalization_173/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????
Cbatch_instance_normalization_173/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_173/moments/varianceMean>batch_instance_normalization_173/moments/SquaredDifference:z:0Lbatch_instance_normalization_173/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ί
$batch_instance_normalization_173/subSubconv2d_221/Conv2D:output:06batch_instance_normalization_173/moments/mean:output:0*
T0*0
_output_shapes
:?????????k
&batch_instance_normalization_173/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Μ
$batch_instance_normalization_173/addAddV2:batch_instance_normalization_173/moments/variance:output:0/batch_instance_normalization_173/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_173/RsqrtRsqrt(batch_instance_normalization_173/add:z:0*
T0*'
_output_shapes
:Ό
$batch_instance_normalization_173/mulMul(batch_instance_normalization_173/sub:z:0*batch_instance_normalization_173/Rsqrt:y:0*
T0*0
_output_shapes
:?????????
Abatch_instance_normalization_173/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      λ
/batch_instance_normalization_173/moments_1/meanMeanconv2d_221/Conv2D:output:0Jbatch_instance_normalization_173/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ό
7batch_instance_normalization_173/moments_1/StopGradientStopGradient8batch_instance_normalization_173/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????κ
<batch_instance_normalization_173/moments_1/SquaredDifferenceSquaredDifferenceconv2d_221/Conv2D:output:0@batch_instance_normalization_173/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????
Ebatch_instance_normalization_173/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_173/moments_1/varianceMean@batch_instance_normalization_173/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_173/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ύ
&batch_instance_normalization_173/sub_1Subconv2d_221/Conv2D:output:08batch_instance_normalization_173/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????m
(batch_instance_normalization_173/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Ϋ
&batch_instance_normalization_173/add_1AddV2<batch_instance_normalization_173/moments_1/variance:output:01batch_instance_normalization_173/add_1/y:output:0*
T0*0
_output_shapes
:?????????
(batch_instance_normalization_173/Rsqrt_1Rsqrt*batch_instance_normalization_173/add_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_173/mul_1Mul*batch_instance_normalization_173/sub_1:z:0,batch_instance_normalization_173/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????₯
/batch_instance_normalization_173/ReadVariableOpReadVariableOp8batch_instance_normalization_173_readvariableop_resource*
_output_shapes	
:*
dtype0Λ
&batch_instance_normalization_173/mul_2Mul7batch_instance_normalization_173/ReadVariableOp:value:0(batch_instance_normalization_173/mul:z:0*
T0*0
_output_shapes
:?????????§
1batch_instance_normalization_173/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_173_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_173/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Α
&batch_instance_normalization_173/sub_2Sub1batch_instance_normalization_173/sub_2/x:output:09batch_instance_normalization_173/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:ΐ
&batch_instance_normalization_173/mul_3Mul*batch_instance_normalization_173/sub_2:z:0*batch_instance_normalization_173/mul_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_173/add_2AddV2*batch_instance_normalization_173/mul_2:z:0*batch_instance_normalization_173/mul_3:z:0*
T0*0
_output_shapes
:?????????±
5batch_instance_normalization_173/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_173_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Σ
&batch_instance_normalization_173/mul_4Mul*batch_instance_normalization_173/add_2:z:0=batch_instance_normalization_173/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????±
5batch_instance_normalization_173/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_173_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Υ
&batch_instance_normalization_173/add_3AddV2*batch_instance_normalization_173/mul_4:z:0=batch_instance_normalization_173/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v
LeakyRelu_2	LeakyRelu*batch_instance_normalization_173/add_3:z:0*0
_output_shapes
:?????????
zero_padding2d_18/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
zero_padding2d_18/PadPadLeakyRelu_2:activations:0'zero_padding2d_18/Pad/paddings:output:0*
T0*0
_output_shapes
:?????????
 conv2d_222/Conv2D/ReadVariableOpReadVariableOp)conv2d_222_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ι
conv2d_222/Conv2DConv2Dzero_padding2d_18/Pad:output:0(conv2d_222/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

?batch_instance_normalization_174/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ή
-batch_instance_normalization_174/moments/meanMeanconv2d_222/Conv2D:output:0Hbatch_instance_normalization_174/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(―
5batch_instance_normalization_174/moments/StopGradientStopGradient6batch_instance_normalization_174/moments/mean:output:0*
T0*'
_output_shapes
:ζ
:batch_instance_normalization_174/moments/SquaredDifferenceSquaredDifferenceconv2d_222/Conv2D:output:0>batch_instance_normalization_174/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????
Cbatch_instance_normalization_174/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_174/moments/varianceMean>batch_instance_normalization_174/moments/SquaredDifference:z:0Lbatch_instance_normalization_174/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ί
$batch_instance_normalization_174/subSubconv2d_222/Conv2D:output:06batch_instance_normalization_174/moments/mean:output:0*
T0*0
_output_shapes
:?????????k
&batch_instance_normalization_174/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Μ
$batch_instance_normalization_174/addAddV2:batch_instance_normalization_174/moments/variance:output:0/batch_instance_normalization_174/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_174/RsqrtRsqrt(batch_instance_normalization_174/add:z:0*
T0*'
_output_shapes
:Ό
$batch_instance_normalization_174/mulMul(batch_instance_normalization_174/sub:z:0*batch_instance_normalization_174/Rsqrt:y:0*
T0*0
_output_shapes
:?????????
Abatch_instance_normalization_174/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      λ
/batch_instance_normalization_174/moments_1/meanMeanconv2d_222/Conv2D:output:0Jbatch_instance_normalization_174/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ό
7batch_instance_normalization_174/moments_1/StopGradientStopGradient8batch_instance_normalization_174/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????κ
<batch_instance_normalization_174/moments_1/SquaredDifferenceSquaredDifferenceconv2d_222/Conv2D:output:0@batch_instance_normalization_174/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????
Ebatch_instance_normalization_174/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_174/moments_1/varianceMean@batch_instance_normalization_174/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_174/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ύ
&batch_instance_normalization_174/sub_1Subconv2d_222/Conv2D:output:08batch_instance_normalization_174/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????m
(batch_instance_normalization_174/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Ϋ
&batch_instance_normalization_174/add_1AddV2<batch_instance_normalization_174/moments_1/variance:output:01batch_instance_normalization_174/add_1/y:output:0*
T0*0
_output_shapes
:?????????
(batch_instance_normalization_174/Rsqrt_1Rsqrt*batch_instance_normalization_174/add_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_174/mul_1Mul*batch_instance_normalization_174/sub_1:z:0,batch_instance_normalization_174/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????₯
/batch_instance_normalization_174/ReadVariableOpReadVariableOp8batch_instance_normalization_174_readvariableop_resource*
_output_shapes	
:*
dtype0Λ
&batch_instance_normalization_174/mul_2Mul7batch_instance_normalization_174/ReadVariableOp:value:0(batch_instance_normalization_174/mul:z:0*
T0*0
_output_shapes
:?????????§
1batch_instance_normalization_174/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_174_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_174/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Α
&batch_instance_normalization_174/sub_2Sub1batch_instance_normalization_174/sub_2/x:output:09batch_instance_normalization_174/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:ΐ
&batch_instance_normalization_174/mul_3Mul*batch_instance_normalization_174/sub_2:z:0*batch_instance_normalization_174/mul_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_174/add_2AddV2*batch_instance_normalization_174/mul_2:z:0*batch_instance_normalization_174/mul_3:z:0*
T0*0
_output_shapes
:?????????±
5batch_instance_normalization_174/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_174_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Σ
&batch_instance_normalization_174/mul_4Mul*batch_instance_normalization_174/add_2:z:0=batch_instance_normalization_174/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????±
5batch_instance_normalization_174/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_174_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Υ
&batch_instance_normalization_174/add_3AddV2*batch_instance_normalization_174/mul_4:z:0=batch_instance_normalization_174/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v
LeakyRelu_3	LeakyRelu*batch_instance_normalization_174/add_3:z:0*0
_output_shapes
:?????????
zero_padding2d_19/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
zero_padding2d_19/PadPadLeakyRelu_3:activations:0'zero_padding2d_19/Pad/paddings:output:0*
T0*0
_output_shapes
:?????????
 conv2d_223/Conv2D/ReadVariableOpReadVariableOp)conv2d_223_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Θ
conv2d_223/Conv2DConv2Dzero_padding2d_19/Pad:output:0(conv2d_223/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides

!conv2d_223/BiasAdd/ReadVariableOpReadVariableOp*conv2d_223_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_223/BiasAddBiasAddconv2d_223/Conv2D:output:0)conv2d_223/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????r
IdentityIdentityconv2d_223/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????
NoOpNoOp0^batch_instance_normalization_172/ReadVariableOp2^batch_instance_normalization_172/ReadVariableOp_16^batch_instance_normalization_172/add_3/ReadVariableOp6^batch_instance_normalization_172/mul_4/ReadVariableOp0^batch_instance_normalization_173/ReadVariableOp2^batch_instance_normalization_173/ReadVariableOp_16^batch_instance_normalization_173/add_3/ReadVariableOp6^batch_instance_normalization_173/mul_4/ReadVariableOp0^batch_instance_normalization_174/ReadVariableOp2^batch_instance_normalization_174/ReadVariableOp_16^batch_instance_normalization_174/add_3/ReadVariableOp6^batch_instance_normalization_174/mul_4/ReadVariableOp!^conv2d_219/Conv2D/ReadVariableOp!^conv2d_220/Conv2D/ReadVariableOp!^conv2d_221/Conv2D/ReadVariableOp!^conv2d_222/Conv2D/ReadVariableOp"^conv2d_223/BiasAdd/ReadVariableOp!^conv2d_223/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 2b
/batch_instance_normalization_172/ReadVariableOp/batch_instance_normalization_172/ReadVariableOp2f
1batch_instance_normalization_172/ReadVariableOp_11batch_instance_normalization_172/ReadVariableOp_12n
5batch_instance_normalization_172/add_3/ReadVariableOp5batch_instance_normalization_172/add_3/ReadVariableOp2n
5batch_instance_normalization_172/mul_4/ReadVariableOp5batch_instance_normalization_172/mul_4/ReadVariableOp2b
/batch_instance_normalization_173/ReadVariableOp/batch_instance_normalization_173/ReadVariableOp2f
1batch_instance_normalization_173/ReadVariableOp_11batch_instance_normalization_173/ReadVariableOp_12n
5batch_instance_normalization_173/add_3/ReadVariableOp5batch_instance_normalization_173/add_3/ReadVariableOp2n
5batch_instance_normalization_173/mul_4/ReadVariableOp5batch_instance_normalization_173/mul_4/ReadVariableOp2b
/batch_instance_normalization_174/ReadVariableOp/batch_instance_normalization_174/ReadVariableOp2f
1batch_instance_normalization_174/ReadVariableOp_11batch_instance_normalization_174/ReadVariableOp_12n
5batch_instance_normalization_174/add_3/ReadVariableOp5batch_instance_normalization_174/add_3/ReadVariableOp2n
5batch_instance_normalization_174/mul_4/ReadVariableOp5batch_instance_normalization_174/mul_4/ReadVariableOp2D
 conv2d_219/Conv2D/ReadVariableOp conv2d_219/Conv2D/ReadVariableOp2D
 conv2d_220/Conv2D/ReadVariableOp conv2d_220/Conv2D/ReadVariableOp2D
 conv2d_221/Conv2D/ReadVariableOp conv2d_221/Conv2D/ReadVariableOp2D
 conv2d_222/Conv2D/ReadVariableOp conv2d_222/Conv2D/ReadVariableOp2F
!conv2d_223/BiasAdd/ReadVariableOp!conv2d_223/BiasAdd/ReadVariableOp2D
 conv2d_223/Conv2D/ReadVariableOp conv2d_223/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/2
Π

-__inference_conv2d_220_layer_call_fn_56549310

inputs"
unknown:@
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56548297x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  `
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
·$
Ξ
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56549368
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identity’ReadVariableOp’ReadVariableOp_1’add_3/ReadVariableOp’mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:?????????  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:?????????  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ά
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:?????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:?????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:?????????  e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:?????????  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:?????????  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:?????????  

_user_specified_namex
¦
Ί
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56548297

inputs9
conv2d_readvariableop_resource:@
identity’Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????  ^
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
π

#__inference__wrapped_model_56548234
input_1
input_2
input_3S
9face_d_region_4_conv2d_219_conv2d_readvariableop_resource:@T
9face_d_region_4_conv2d_220_conv2d_readvariableop_resource:@W
Hface_d_region_4_batch_instance_normalization_172_readvariableop_resource:	]
Nface_d_region_4_batch_instance_normalization_172_mul_4_readvariableop_resource:	]
Nface_d_region_4_batch_instance_normalization_172_add_3_readvariableop_resource:	U
9face_d_region_4_conv2d_221_conv2d_readvariableop_resource:W
Hface_d_region_4_batch_instance_normalization_173_readvariableop_resource:	]
Nface_d_region_4_batch_instance_normalization_173_mul_4_readvariableop_resource:	]
Nface_d_region_4_batch_instance_normalization_173_add_3_readvariableop_resource:	U
9face_d_region_4_conv2d_222_conv2d_readvariableop_resource:W
Hface_d_region_4_batch_instance_normalization_174_readvariableop_resource:	]
Nface_d_region_4_batch_instance_normalization_174_mul_4_readvariableop_resource:	]
Nface_d_region_4_batch_instance_normalization_174_add_3_readvariableop_resource:	T
9face_d_region_4_conv2d_223_conv2d_readvariableop_resource:H
:face_d_region_4_conv2d_223_biasadd_readvariableop_resource:
identity’?face_d_region_4/batch_instance_normalization_172/ReadVariableOp’Aface_d_region_4/batch_instance_normalization_172/ReadVariableOp_1’Eface_d_region_4/batch_instance_normalization_172/add_3/ReadVariableOp’Eface_d_region_4/batch_instance_normalization_172/mul_4/ReadVariableOp’?face_d_region_4/batch_instance_normalization_173/ReadVariableOp’Aface_d_region_4/batch_instance_normalization_173/ReadVariableOp_1’Eface_d_region_4/batch_instance_normalization_173/add_3/ReadVariableOp’Eface_d_region_4/batch_instance_normalization_173/mul_4/ReadVariableOp’?face_d_region_4/batch_instance_normalization_174/ReadVariableOp’Aface_d_region_4/batch_instance_normalization_174/ReadVariableOp_1’Eface_d_region_4/batch_instance_normalization_174/add_3/ReadVariableOp’Eface_d_region_4/batch_instance_normalization_174/mul_4/ReadVariableOp’0face_d_region_4/conv2d_219/Conv2D/ReadVariableOp’0face_d_region_4/conv2d_220/Conv2D/ReadVariableOp’0face_d_region_4/conv2d_221/Conv2D/ReadVariableOp’0face_d_region_4/conv2d_222/Conv2D/ReadVariableOp’1face_d_region_4/conv2d_223/BiasAdd/ReadVariableOp’0face_d_region_4/conv2d_223/Conv2D/ReadVariableOpe
 face_d_region_4/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C
face_d_region_4/lambda/truedivRealDivinput_2)face_d_region_4/lambda/truediv/y:output:0*
T0*1
_output_shapes
:?????????a
face_d_region_4/lambda/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¨
face_d_region_4/lambda/subSub%face_d_region_4/lambda/sub/x:output:0"face_d_region_4/lambda/truediv:z:0*
T0*1
_output_shapes
:?????????
#face_d_region_4/lambda/multiply/mulMulinput_3face_d_region_4/lambda/sub:z:0*
T0*1
_output_shapes
:?????????
%face_d_region_4/lambda/multiply_1/mulMulinput_1"face_d_region_4/lambda/truediv:z:0*
T0*1
_output_shapes
:?????????·
face_d_region_4/lambda/add/addAddV2'face_d_region_4/lambda/multiply/mul:z:0)face_d_region_4/lambda/multiply_1/mul:z:0*
T0*1
_output_shapes
:?????????²
0face_d_region_4/conv2d_219/Conv2D/ReadVariableOpReadVariableOp9face_d_region_4_conv2d_219_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0λ
!face_d_region_4/conv2d_219/Conv2DConv2D"face_d_region_4/lambda/add/add:z:08face_d_region_4/conv2d_219/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides

face_d_region_4/LeakyRelu	LeakyRelu*face_d_region_4/conv2d_219/Conv2D:output:0*/
_output_shapes
:?????????@@@³
0face_d_region_4/conv2d_220/Conv2D/ReadVariableOpReadVariableOp9face_d_region_4_conv2d_220_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ρ
!face_d_region_4/conv2d_220/Conv2DConv2D'face_d_region_4/LeakyRelu:activations:08face_d_region_4/conv2d_220/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
€
Oface_d_region_4/batch_instance_normalization_172/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
=face_d_region_4/batch_instance_normalization_172/moments/meanMean*face_d_region_4/conv2d_220/Conv2D:output:0Xface_d_region_4/batch_instance_normalization_172/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ο
Eface_d_region_4/batch_instance_normalization_172/moments/StopGradientStopGradientFface_d_region_4/batch_instance_normalization_172/moments/mean:output:0*
T0*'
_output_shapes
:
Jface_d_region_4/batch_instance_normalization_172/moments/SquaredDifferenceSquaredDifference*face_d_region_4/conv2d_220/Conv2D:output:0Nface_d_region_4/batch_instance_normalization_172/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ¨
Sface_d_region_4/batch_instance_normalization_172/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ί
Aface_d_region_4/batch_instance_normalization_172/moments/varianceMeanNface_d_region_4/batch_instance_normalization_172/moments/SquaredDifference:z:0\face_d_region_4/batch_instance_normalization_172/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(κ
4face_d_region_4/batch_instance_normalization_172/subSub*face_d_region_4/conv2d_220/Conv2D:output:0Fface_d_region_4/batch_instance_normalization_172/moments/mean:output:0*
T0*0
_output_shapes
:?????????  {
6face_d_region_4/batch_instance_normalization_172/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7ό
4face_d_region_4/batch_instance_normalization_172/addAddV2Jface_d_region_4/batch_instance_normalization_172/moments/variance:output:0?face_d_region_4/batch_instance_normalization_172/add/y:output:0*
T0*'
_output_shapes
:«
6face_d_region_4/batch_instance_normalization_172/RsqrtRsqrt8face_d_region_4/batch_instance_normalization_172/add:z:0*
T0*'
_output_shapes
:μ
4face_d_region_4/batch_instance_normalization_172/mulMul8face_d_region_4/batch_instance_normalization_172/sub:z:0:face_d_region_4/batch_instance_normalization_172/Rsqrt:y:0*
T0*0
_output_shapes
:?????????  ’
Qface_d_region_4/batch_instance_normalization_172/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
?face_d_region_4/batch_instance_normalization_172/moments_1/meanMean*face_d_region_4/conv2d_220/Conv2D:output:0Zface_d_region_4/batch_instance_normalization_172/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(ά
Gface_d_region_4/batch_instance_normalization_172/moments_1/StopGradientStopGradientHface_d_region_4/batch_instance_normalization_172/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????
Lface_d_region_4/batch_instance_normalization_172/moments_1/SquaredDifferenceSquaredDifference*face_d_region_4/conv2d_220/Conv2D:output:0Pface_d_region_4/batch_instance_normalization_172/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  ¦
Uface_d_region_4/batch_instance_normalization_172/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ι
Cface_d_region_4/batch_instance_normalization_172/moments_1/varianceMeanPface_d_region_4/batch_instance_normalization_172/moments_1/SquaredDifference:z:0^face_d_region_4/batch_instance_normalization_172/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(ξ
6face_d_region_4/batch_instance_normalization_172/sub_1Sub*face_d_region_4/conv2d_220/Conv2D:output:0Hface_d_region_4/batch_instance_normalization_172/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????  }
8face_d_region_4/batch_instance_normalization_172/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7
6face_d_region_4/batch_instance_normalization_172/add_1AddV2Lface_d_region_4/batch_instance_normalization_172/moments_1/variance:output:0Aface_d_region_4/batch_instance_normalization_172/add_1/y:output:0*
T0*0
_output_shapes
:?????????Έ
8face_d_region_4/batch_instance_normalization_172/Rsqrt_1Rsqrt:face_d_region_4/batch_instance_normalization_172/add_1:z:0*
T0*0
_output_shapes
:?????????ς
6face_d_region_4/batch_instance_normalization_172/mul_1Mul:face_d_region_4/batch_instance_normalization_172/sub_1:z:0<face_d_region_4/batch_instance_normalization_172/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  Ε
?face_d_region_4/batch_instance_normalization_172/ReadVariableOpReadVariableOpHface_d_region_4_batch_instance_normalization_172_readvariableop_resource*
_output_shapes	
:*
dtype0ϋ
6face_d_region_4/batch_instance_normalization_172/mul_2MulGface_d_region_4/batch_instance_normalization_172/ReadVariableOp:value:08face_d_region_4/batch_instance_normalization_172/mul:z:0*
T0*0
_output_shapes
:?????????  Η
Aface_d_region_4/batch_instance_normalization_172/ReadVariableOp_1ReadVariableOpHface_d_region_4_batch_instance_normalization_172_readvariableop_resource*
_output_shapes	
:*
dtype0}
8face_d_region_4/batch_instance_normalization_172/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ρ
6face_d_region_4/batch_instance_normalization_172/sub_2SubAface_d_region_4/batch_instance_normalization_172/sub_2/x:output:0Iface_d_region_4/batch_instance_normalization_172/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:π
6face_d_region_4/batch_instance_normalization_172/mul_3Mul:face_d_region_4/batch_instance_normalization_172/sub_2:z:0:face_d_region_4/batch_instance_normalization_172/mul_1:z:0*
T0*0
_output_shapes
:?????????  ς
6face_d_region_4/batch_instance_normalization_172/add_2AddV2:face_d_region_4/batch_instance_normalization_172/mul_2:z:0:face_d_region_4/batch_instance_normalization_172/mul_3:z:0*
T0*0
_output_shapes
:?????????  Ρ
Eface_d_region_4/batch_instance_normalization_172/mul_4/ReadVariableOpReadVariableOpNface_d_region_4_batch_instance_normalization_172_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0
6face_d_region_4/batch_instance_normalization_172/mul_4Mul:face_d_region_4/batch_instance_normalization_172/add_2:z:0Mface_d_region_4/batch_instance_normalization_172/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  Ρ
Eface_d_region_4/batch_instance_normalization_172/add_3/ReadVariableOpReadVariableOpNface_d_region_4_batch_instance_normalization_172_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0
6face_d_region_4/batch_instance_normalization_172/add_3AddV2:face_d_region_4/batch_instance_normalization_172/mul_4:z:0Mface_d_region_4/batch_instance_normalization_172/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  
face_d_region_4/LeakyRelu_1	LeakyRelu:face_d_region_4/batch_instance_normalization_172/add_3:z:0*0
_output_shapes
:?????????  ΄
0face_d_region_4/conv2d_221/Conv2D/ReadVariableOpReadVariableOp9face_d_region_4_conv2d_221_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0σ
!face_d_region_4/conv2d_221/Conv2DConv2D)face_d_region_4/LeakyRelu_1:activations:08face_d_region_4/conv2d_221/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
€
Oface_d_region_4/batch_instance_normalization_173/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
=face_d_region_4/batch_instance_normalization_173/moments/meanMean*face_d_region_4/conv2d_221/Conv2D:output:0Xface_d_region_4/batch_instance_normalization_173/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ο
Eface_d_region_4/batch_instance_normalization_173/moments/StopGradientStopGradientFface_d_region_4/batch_instance_normalization_173/moments/mean:output:0*
T0*'
_output_shapes
:
Jface_d_region_4/batch_instance_normalization_173/moments/SquaredDifferenceSquaredDifference*face_d_region_4/conv2d_221/Conv2D:output:0Nface_d_region_4/batch_instance_normalization_173/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????¨
Sface_d_region_4/batch_instance_normalization_173/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ί
Aface_d_region_4/batch_instance_normalization_173/moments/varianceMeanNface_d_region_4/batch_instance_normalization_173/moments/SquaredDifference:z:0\face_d_region_4/batch_instance_normalization_173/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(κ
4face_d_region_4/batch_instance_normalization_173/subSub*face_d_region_4/conv2d_221/Conv2D:output:0Fface_d_region_4/batch_instance_normalization_173/moments/mean:output:0*
T0*0
_output_shapes
:?????????{
6face_d_region_4/batch_instance_normalization_173/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7ό
4face_d_region_4/batch_instance_normalization_173/addAddV2Jface_d_region_4/batch_instance_normalization_173/moments/variance:output:0?face_d_region_4/batch_instance_normalization_173/add/y:output:0*
T0*'
_output_shapes
:«
6face_d_region_4/batch_instance_normalization_173/RsqrtRsqrt8face_d_region_4/batch_instance_normalization_173/add:z:0*
T0*'
_output_shapes
:μ
4face_d_region_4/batch_instance_normalization_173/mulMul8face_d_region_4/batch_instance_normalization_173/sub:z:0:face_d_region_4/batch_instance_normalization_173/Rsqrt:y:0*
T0*0
_output_shapes
:?????????’
Qface_d_region_4/batch_instance_normalization_173/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
?face_d_region_4/batch_instance_normalization_173/moments_1/meanMean*face_d_region_4/conv2d_221/Conv2D:output:0Zface_d_region_4/batch_instance_normalization_173/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(ά
Gface_d_region_4/batch_instance_normalization_173/moments_1/StopGradientStopGradientHface_d_region_4/batch_instance_normalization_173/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????
Lface_d_region_4/batch_instance_normalization_173/moments_1/SquaredDifferenceSquaredDifference*face_d_region_4/conv2d_221/Conv2D:output:0Pface_d_region_4/batch_instance_normalization_173/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????¦
Uface_d_region_4/batch_instance_normalization_173/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ι
Cface_d_region_4/batch_instance_normalization_173/moments_1/varianceMeanPface_d_region_4/batch_instance_normalization_173/moments_1/SquaredDifference:z:0^face_d_region_4/batch_instance_normalization_173/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(ξ
6face_d_region_4/batch_instance_normalization_173/sub_1Sub*face_d_region_4/conv2d_221/Conv2D:output:0Hface_d_region_4/batch_instance_normalization_173/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????}
8face_d_region_4/batch_instance_normalization_173/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7
6face_d_region_4/batch_instance_normalization_173/add_1AddV2Lface_d_region_4/batch_instance_normalization_173/moments_1/variance:output:0Aface_d_region_4/batch_instance_normalization_173/add_1/y:output:0*
T0*0
_output_shapes
:?????????Έ
8face_d_region_4/batch_instance_normalization_173/Rsqrt_1Rsqrt:face_d_region_4/batch_instance_normalization_173/add_1:z:0*
T0*0
_output_shapes
:?????????ς
6face_d_region_4/batch_instance_normalization_173/mul_1Mul:face_d_region_4/batch_instance_normalization_173/sub_1:z:0<face_d_region_4/batch_instance_normalization_173/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????Ε
?face_d_region_4/batch_instance_normalization_173/ReadVariableOpReadVariableOpHface_d_region_4_batch_instance_normalization_173_readvariableop_resource*
_output_shapes	
:*
dtype0ϋ
6face_d_region_4/batch_instance_normalization_173/mul_2MulGface_d_region_4/batch_instance_normalization_173/ReadVariableOp:value:08face_d_region_4/batch_instance_normalization_173/mul:z:0*
T0*0
_output_shapes
:?????????Η
Aface_d_region_4/batch_instance_normalization_173/ReadVariableOp_1ReadVariableOpHface_d_region_4_batch_instance_normalization_173_readvariableop_resource*
_output_shapes	
:*
dtype0}
8face_d_region_4/batch_instance_normalization_173/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ρ
6face_d_region_4/batch_instance_normalization_173/sub_2SubAface_d_region_4/batch_instance_normalization_173/sub_2/x:output:0Iface_d_region_4/batch_instance_normalization_173/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:π
6face_d_region_4/batch_instance_normalization_173/mul_3Mul:face_d_region_4/batch_instance_normalization_173/sub_2:z:0:face_d_region_4/batch_instance_normalization_173/mul_1:z:0*
T0*0
_output_shapes
:?????????ς
6face_d_region_4/batch_instance_normalization_173/add_2AddV2:face_d_region_4/batch_instance_normalization_173/mul_2:z:0:face_d_region_4/batch_instance_normalization_173/mul_3:z:0*
T0*0
_output_shapes
:?????????Ρ
Eface_d_region_4/batch_instance_normalization_173/mul_4/ReadVariableOpReadVariableOpNface_d_region_4_batch_instance_normalization_173_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0
6face_d_region_4/batch_instance_normalization_173/mul_4Mul:face_d_region_4/batch_instance_normalization_173/add_2:z:0Mface_d_region_4/batch_instance_normalization_173/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Ρ
Eface_d_region_4/batch_instance_normalization_173/add_3/ReadVariableOpReadVariableOpNface_d_region_4_batch_instance_normalization_173_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0
6face_d_region_4/batch_instance_normalization_173/add_3AddV2:face_d_region_4/batch_instance_normalization_173/mul_4:z:0Mface_d_region_4/batch_instance_normalization_173/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
face_d_region_4/LeakyRelu_2	LeakyRelu:face_d_region_4/batch_instance_normalization_173/add_3:z:0*0
_output_shapes
:?????????
.face_d_region_4/zero_padding2d_18/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             Λ
%face_d_region_4/zero_padding2d_18/PadPad)face_d_region_4/LeakyRelu_2:activations:07face_d_region_4/zero_padding2d_18/Pad/paddings:output:0*
T0*0
_output_shapes
:?????????΄
0face_d_region_4/conv2d_222/Conv2D/ReadVariableOpReadVariableOp9face_d_region_4_conv2d_222_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ω
!face_d_region_4/conv2d_222/Conv2DConv2D.face_d_region_4/zero_padding2d_18/Pad:output:08face_d_region_4/conv2d_222/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
€
Oface_d_region_4/batch_instance_normalization_174/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
=face_d_region_4/batch_instance_normalization_174/moments/meanMean*face_d_region_4/conv2d_222/Conv2D:output:0Xface_d_region_4/batch_instance_normalization_174/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ο
Eface_d_region_4/batch_instance_normalization_174/moments/StopGradientStopGradientFface_d_region_4/batch_instance_normalization_174/moments/mean:output:0*
T0*'
_output_shapes
:
Jface_d_region_4/batch_instance_normalization_174/moments/SquaredDifferenceSquaredDifference*face_d_region_4/conv2d_222/Conv2D:output:0Nface_d_region_4/batch_instance_normalization_174/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????¨
Sface_d_region_4/batch_instance_normalization_174/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ί
Aface_d_region_4/batch_instance_normalization_174/moments/varianceMeanNface_d_region_4/batch_instance_normalization_174/moments/SquaredDifference:z:0\face_d_region_4/batch_instance_normalization_174/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(κ
4face_d_region_4/batch_instance_normalization_174/subSub*face_d_region_4/conv2d_222/Conv2D:output:0Fface_d_region_4/batch_instance_normalization_174/moments/mean:output:0*
T0*0
_output_shapes
:?????????{
6face_d_region_4/batch_instance_normalization_174/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7ό
4face_d_region_4/batch_instance_normalization_174/addAddV2Jface_d_region_4/batch_instance_normalization_174/moments/variance:output:0?face_d_region_4/batch_instance_normalization_174/add/y:output:0*
T0*'
_output_shapes
:«
6face_d_region_4/batch_instance_normalization_174/RsqrtRsqrt8face_d_region_4/batch_instance_normalization_174/add:z:0*
T0*'
_output_shapes
:μ
4face_d_region_4/batch_instance_normalization_174/mulMul8face_d_region_4/batch_instance_normalization_174/sub:z:0:face_d_region_4/batch_instance_normalization_174/Rsqrt:y:0*
T0*0
_output_shapes
:?????????’
Qface_d_region_4/batch_instance_normalization_174/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
?face_d_region_4/batch_instance_normalization_174/moments_1/meanMean*face_d_region_4/conv2d_222/Conv2D:output:0Zface_d_region_4/batch_instance_normalization_174/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(ά
Gface_d_region_4/batch_instance_normalization_174/moments_1/StopGradientStopGradientHface_d_region_4/batch_instance_normalization_174/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????
Lface_d_region_4/batch_instance_normalization_174/moments_1/SquaredDifferenceSquaredDifference*face_d_region_4/conv2d_222/Conv2D:output:0Pface_d_region_4/batch_instance_normalization_174/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????¦
Uface_d_region_4/batch_instance_normalization_174/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ι
Cface_d_region_4/batch_instance_normalization_174/moments_1/varianceMeanPface_d_region_4/batch_instance_normalization_174/moments_1/SquaredDifference:z:0^face_d_region_4/batch_instance_normalization_174/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(ξ
6face_d_region_4/batch_instance_normalization_174/sub_1Sub*face_d_region_4/conv2d_222/Conv2D:output:0Hface_d_region_4/batch_instance_normalization_174/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????}
8face_d_region_4/batch_instance_normalization_174/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7
6face_d_region_4/batch_instance_normalization_174/add_1AddV2Lface_d_region_4/batch_instance_normalization_174/moments_1/variance:output:0Aface_d_region_4/batch_instance_normalization_174/add_1/y:output:0*
T0*0
_output_shapes
:?????????Έ
8face_d_region_4/batch_instance_normalization_174/Rsqrt_1Rsqrt:face_d_region_4/batch_instance_normalization_174/add_1:z:0*
T0*0
_output_shapes
:?????????ς
6face_d_region_4/batch_instance_normalization_174/mul_1Mul:face_d_region_4/batch_instance_normalization_174/sub_1:z:0<face_d_region_4/batch_instance_normalization_174/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????Ε
?face_d_region_4/batch_instance_normalization_174/ReadVariableOpReadVariableOpHface_d_region_4_batch_instance_normalization_174_readvariableop_resource*
_output_shapes	
:*
dtype0ϋ
6face_d_region_4/batch_instance_normalization_174/mul_2MulGface_d_region_4/batch_instance_normalization_174/ReadVariableOp:value:08face_d_region_4/batch_instance_normalization_174/mul:z:0*
T0*0
_output_shapes
:?????????Η
Aface_d_region_4/batch_instance_normalization_174/ReadVariableOp_1ReadVariableOpHface_d_region_4_batch_instance_normalization_174_readvariableop_resource*
_output_shapes	
:*
dtype0}
8face_d_region_4/batch_instance_normalization_174/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ρ
6face_d_region_4/batch_instance_normalization_174/sub_2SubAface_d_region_4/batch_instance_normalization_174/sub_2/x:output:0Iface_d_region_4/batch_instance_normalization_174/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:π
6face_d_region_4/batch_instance_normalization_174/mul_3Mul:face_d_region_4/batch_instance_normalization_174/sub_2:z:0:face_d_region_4/batch_instance_normalization_174/mul_1:z:0*
T0*0
_output_shapes
:?????????ς
6face_d_region_4/batch_instance_normalization_174/add_2AddV2:face_d_region_4/batch_instance_normalization_174/mul_2:z:0:face_d_region_4/batch_instance_normalization_174/mul_3:z:0*
T0*0
_output_shapes
:?????????Ρ
Eface_d_region_4/batch_instance_normalization_174/mul_4/ReadVariableOpReadVariableOpNface_d_region_4_batch_instance_normalization_174_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0
6face_d_region_4/batch_instance_normalization_174/mul_4Mul:face_d_region_4/batch_instance_normalization_174/add_2:z:0Mface_d_region_4/batch_instance_normalization_174/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????Ρ
Eface_d_region_4/batch_instance_normalization_174/add_3/ReadVariableOpReadVariableOpNface_d_region_4_batch_instance_normalization_174_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0
6face_d_region_4/batch_instance_normalization_174/add_3AddV2:face_d_region_4/batch_instance_normalization_174/mul_4:z:0Mface_d_region_4/batch_instance_normalization_174/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
face_d_region_4/LeakyRelu_3	LeakyRelu:face_d_region_4/batch_instance_normalization_174/add_3:z:0*0
_output_shapes
:?????????
.face_d_region_4/zero_padding2d_19/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             Λ
%face_d_region_4/zero_padding2d_19/PadPad)face_d_region_4/LeakyRelu_3:activations:07face_d_region_4/zero_padding2d_19/Pad/paddings:output:0*
T0*0
_output_shapes
:?????????³
0face_d_region_4/conv2d_223/Conv2D/ReadVariableOpReadVariableOp9face_d_region_4_conv2d_223_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0ψ
!face_d_region_4/conv2d_223/Conv2DConv2D.face_d_region_4/zero_padding2d_19/Pad:output:08face_d_region_4/conv2d_223/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
¨
1face_d_region_4/conv2d_223/BiasAdd/ReadVariableOpReadVariableOp:face_d_region_4_conv2d_223_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ξ
"face_d_region_4/conv2d_223/BiasAddBiasAdd*face_d_region_4/conv2d_223/Conv2D:output:09face_d_region_4/conv2d_223/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????
IdentityIdentity+face_d_region_4/conv2d_223/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????»	
NoOpNoOp@^face_d_region_4/batch_instance_normalization_172/ReadVariableOpB^face_d_region_4/batch_instance_normalization_172/ReadVariableOp_1F^face_d_region_4/batch_instance_normalization_172/add_3/ReadVariableOpF^face_d_region_4/batch_instance_normalization_172/mul_4/ReadVariableOp@^face_d_region_4/batch_instance_normalization_173/ReadVariableOpB^face_d_region_4/batch_instance_normalization_173/ReadVariableOp_1F^face_d_region_4/batch_instance_normalization_173/add_3/ReadVariableOpF^face_d_region_4/batch_instance_normalization_173/mul_4/ReadVariableOp@^face_d_region_4/batch_instance_normalization_174/ReadVariableOpB^face_d_region_4/batch_instance_normalization_174/ReadVariableOp_1F^face_d_region_4/batch_instance_normalization_174/add_3/ReadVariableOpF^face_d_region_4/batch_instance_normalization_174/mul_4/ReadVariableOp1^face_d_region_4/conv2d_219/Conv2D/ReadVariableOp1^face_d_region_4/conv2d_220/Conv2D/ReadVariableOp1^face_d_region_4/conv2d_221/Conv2D/ReadVariableOp1^face_d_region_4/conv2d_222/Conv2D/ReadVariableOp2^face_d_region_4/conv2d_223/BiasAdd/ReadVariableOp1^face_d_region_4/conv2d_223/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 2
?face_d_region_4/batch_instance_normalization_172/ReadVariableOp?face_d_region_4/batch_instance_normalization_172/ReadVariableOp2
Aface_d_region_4/batch_instance_normalization_172/ReadVariableOp_1Aface_d_region_4/batch_instance_normalization_172/ReadVariableOp_12
Eface_d_region_4/batch_instance_normalization_172/add_3/ReadVariableOpEface_d_region_4/batch_instance_normalization_172/add_3/ReadVariableOp2
Eface_d_region_4/batch_instance_normalization_172/mul_4/ReadVariableOpEface_d_region_4/batch_instance_normalization_172/mul_4/ReadVariableOp2
?face_d_region_4/batch_instance_normalization_173/ReadVariableOp?face_d_region_4/batch_instance_normalization_173/ReadVariableOp2
Aface_d_region_4/batch_instance_normalization_173/ReadVariableOp_1Aface_d_region_4/batch_instance_normalization_173/ReadVariableOp_12
Eface_d_region_4/batch_instance_normalization_173/add_3/ReadVariableOpEface_d_region_4/batch_instance_normalization_173/add_3/ReadVariableOp2
Eface_d_region_4/batch_instance_normalization_173/mul_4/ReadVariableOpEface_d_region_4/batch_instance_normalization_173/mul_4/ReadVariableOp2
?face_d_region_4/batch_instance_normalization_174/ReadVariableOp?face_d_region_4/batch_instance_normalization_174/ReadVariableOp2
Aface_d_region_4/batch_instance_normalization_174/ReadVariableOp_1Aface_d_region_4/batch_instance_normalization_174/ReadVariableOp_12
Eface_d_region_4/batch_instance_normalization_174/add_3/ReadVariableOpEface_d_region_4/batch_instance_normalization_174/add_3/ReadVariableOp2
Eface_d_region_4/batch_instance_normalization_174/mul_4/ReadVariableOpEface_d_region_4/batch_instance_normalization_174/mul_4/ReadVariableOp2d
0face_d_region_4/conv2d_219/Conv2D/ReadVariableOp0face_d_region_4/conv2d_219/Conv2D/ReadVariableOp2d
0face_d_region_4/conv2d_220/Conv2D/ReadVariableOp0face_d_region_4/conv2d_220/Conv2D/ReadVariableOp2d
0face_d_region_4/conv2d_221/Conv2D/ReadVariableOp0face_d_region_4/conv2d_221/Conv2D/ReadVariableOp2d
0face_d_region_4/conv2d_222/Conv2D/ReadVariableOp0face_d_region_4/conv2d_222/Conv2D/ReadVariableOp2f
1face_d_region_4/conv2d_223/BiasAdd/ReadVariableOp1face_d_region_4/conv2d_223/BiasAdd/ReadVariableOp2d
0face_d_region_4/conv2d_223/Conv2D/ReadVariableOp0face_d_region_4/conv2d_223/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:?????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_2:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_3
E


M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548489

inputs
inputs_1
inputs_2-
conv2d_219_56548286:@.
conv2d_220_56548298:@8
)batch_instance_normalization_172_56548342:	8
)batch_instance_normalization_172_56548344:	8
)batch_instance_normalization_172_56548346:	/
conv2d_221_56548358:8
)batch_instance_normalization_173_56548402:	8
)batch_instance_normalization_173_56548404:	8
)batch_instance_normalization_173_56548406:	/
conv2d_222_56548419:8
)batch_instance_normalization_174_56548463:	8
)batch_instance_normalization_174_56548465:	8
)batch_instance_normalization_174_56548467:	.
conv2d_223_56548483:!
conv2d_223_56548485:
identity’8batch_instance_normalization_172/StatefulPartitionedCall’8batch_instance_normalization_173/StatefulPartitionedCall’8batch_instance_normalization_174/StatefulPartitionedCall’"conv2d_219/StatefulPartitionedCall’"conv2d_220/StatefulPartitionedCall’"conv2d_221/StatefulPartitionedCall’"conv2d_222/StatefulPartitionedCall’"conv2d_223/StatefulPartitionedCallU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cz
lambda/truedivRealDivinputs_1lambda/truediv/y:output:0*
T0*1
_output_shapes
:?????????Q
lambda/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x

lambda/subSublambda/sub/x:output:0lambda/truediv:z:0*
T0*1
_output_shapes
:?????????p
lambda/multiply/mulMulinputs_2lambda/sub:z:0*
T0*1
_output_shapes
:?????????t
lambda/multiply_1/mulMulinputslambda/truediv:z:0*
T0*1
_output_shapes
:?????????
lambda/add/addAddV2lambda/multiply/mul:z:0lambda/multiply_1/mul:z:0*
T0*1
_output_shapes
:?????????ώ
"conv2d_219/StatefulPartitionedCallStatefulPartitionedCalllambda/add/add:z:0conv2d_219_56548286*
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56548285t
	LeakyRelu	LeakyRelu+conv2d_219/StatefulPartitionedCall:output:0*/
_output_shapes
:?????????@@@
"conv2d_220/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu:activations:0conv2d_220_56548298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56548297΄
8batch_instance_normalization_172/StatefulPartitionedCallStatefulPartitionedCall+conv2d_220/StatefulPartitionedCall:output:0)batch_instance_normalization_172_56548342)batch_instance_normalization_172_56548344)batch_instance_normalization_172_56548346*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56548341
LeakyRelu_1	LeakyReluAbatch_instance_normalization_172/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????  
"conv2d_221/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_221_56548358*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56548357΄
8batch_instance_normalization_173/StatefulPartitionedCallStatefulPartitionedCall+conv2d_221/StatefulPartitionedCall:output:0)batch_instance_normalization_173_56548402)batch_instance_normalization_173_56548404)batch_instance_normalization_173_56548406*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56548401
LeakyRelu_2	LeakyReluAbatch_instance_normalization_173/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????λ
!zero_padding2d_18/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56548244
"conv2d_222/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_18/PartitionedCall:output:0conv2d_222_56548419*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56548418΄
8batch_instance_normalization_174/StatefulPartitionedCallStatefulPartitionedCall+conv2d_222/StatefulPartitionedCall:output:0)batch_instance_normalization_174_56548463)batch_instance_normalization_174_56548465)batch_instance_normalization_174_56548467*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56548462
LeakyRelu_3	LeakyReluAbatch_instance_normalization_174/StatefulPartitionedCall:output:0*0
_output_shapes
:?????????λ
!zero_padding2d_19/PartitionedCallPartitionedCallLeakyRelu_3:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56548257­
"conv2d_223/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_19/PartitionedCall:output:0conv2d_223_56548483conv2d_223_56548485*
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
GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_223_layer_call_and_return_conditional_losses_56548482
IdentityIdentity+conv2d_223/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????°
NoOpNoOp9^batch_instance_normalization_172/StatefulPartitionedCall9^batch_instance_normalization_173/StatefulPartitionedCall9^batch_instance_normalization_174/StatefulPartitionedCall#^conv2d_219/StatefulPartitionedCall#^conv2d_220/StatefulPartitionedCall#^conv2d_221/StatefulPartitionedCall#^conv2d_222/StatefulPartitionedCall#^conv2d_223/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 2t
8batch_instance_normalization_172/StatefulPartitionedCall8batch_instance_normalization_172/StatefulPartitionedCall2t
8batch_instance_normalization_173/StatefulPartitionedCall8batch_instance_normalization_173/StatefulPartitionedCall2t
8batch_instance_normalization_174/StatefulPartitionedCall8batch_instance_normalization_174/StatefulPartitionedCall2H
"conv2d_219/StatefulPartitionedCall"conv2d_219/StatefulPartitionedCall2H
"conv2d_220/StatefulPartitionedCall"conv2d_220/StatefulPartitionedCall2H
"conv2d_221/StatefulPartitionedCall"conv2d_221/StatefulPartitionedCall2H
"conv2d_222/StatefulPartitionedCall"conv2d_222/StatefulPartitionedCall2H
"conv2d_223/StatefulPartitionedCall"conv2d_223/StatefulPartitionedCall:Y U
1
_output_shapes
:?????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:?????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:?????????
 
_user_specified_nameinputs
φ
Κ
2__inference_face_d_region_4_layer_call_fn_56548522
input_1
input_2
input_3!
unknown:@$
	unknown_0:@
	unknown_1:	
	unknown_2:	
	unknown_3:	%
	unknown_4:
	unknown_5:	
	unknown_6:	
	unknown_7:	%
	unknown_8:
	unknown_9:	

unknown_10:	

unknown_11:	%

unknown_12:

unknown_13:
identity’StatefulPartitionedCall―
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*1
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548489w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:?????????
!
_user_specified_name	input_1:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_2:ZV
1
_output_shapes
:?????????
!
_user_specified_name	input_3
·$
Ξ
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56548401
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identity’ReadVariableOp’ReadVariableOp_1’add_3/ReadVariableOp’mul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(m
moments/StopGradientStopGradientmoments/mean:output:0*
T0*'
_output_shapes
:
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*0
_output_shapes
:?????????w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(_
subSubxmoments/mean:output:0*
T0*0
_output_shapes
:?????????J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7i
addAddV2moments/variance:output:0add/y:output:0*
T0*'
_output_shapes
:I
RsqrtRsqrtadd:z:0*
T0*'
_output_shapes
:Y
mulMulsub:z:0	Rsqrt:y:0*
T0*0
_output_shapes
:?????????q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      Ά
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:?????????L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:?????????V
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:?????????_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:?????????e
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?^
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes	
:]
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*0
_output_shapes
:?????????_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:?????????o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:?????????
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:?????????

_user_specified_namex
¦
Ί
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56549317

inputs9
conv2d_readvariableop_resource:@
identity’Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????  ^
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
ͺ
»
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56549382

inputs:
conv2d_readvariableop_resource:
identity’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  
 
_user_specified_nameinputs
΅
Γ
C__inference_batch_instance_normalization_173_layer_call_fn_56549393
x
unknown:	
	unknown_0:	
	unknown_1:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56548401x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????

_user_specified_namex
΅
Γ
C__inference_batch_instance_normalization_174_layer_call_fn_56549469
x
unknown:	
	unknown_0:	
	unknown_1:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56548462x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":?????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:?????????

_user_specified_namex
ήΩ
­
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56549250
inputs_0
inputs_1
inputs_2C
)conv2d_219_conv2d_readvariableop_resource:@D
)conv2d_220_conv2d_readvariableop_resource:@G
8batch_instance_normalization_172_readvariableop_resource:	M
>batch_instance_normalization_172_mul_4_readvariableop_resource:	M
>batch_instance_normalization_172_add_3_readvariableop_resource:	E
)conv2d_221_conv2d_readvariableop_resource:G
8batch_instance_normalization_173_readvariableop_resource:	M
>batch_instance_normalization_173_mul_4_readvariableop_resource:	M
>batch_instance_normalization_173_add_3_readvariableop_resource:	E
)conv2d_222_conv2d_readvariableop_resource:G
8batch_instance_normalization_174_readvariableop_resource:	M
>batch_instance_normalization_174_mul_4_readvariableop_resource:	M
>batch_instance_normalization_174_add_3_readvariableop_resource:	D
)conv2d_223_conv2d_readvariableop_resource:8
*conv2d_223_biasadd_readvariableop_resource:
identity’/batch_instance_normalization_172/ReadVariableOp’1batch_instance_normalization_172/ReadVariableOp_1’5batch_instance_normalization_172/add_3/ReadVariableOp’5batch_instance_normalization_172/mul_4/ReadVariableOp’/batch_instance_normalization_173/ReadVariableOp’1batch_instance_normalization_173/ReadVariableOp_1’5batch_instance_normalization_173/add_3/ReadVariableOp’5batch_instance_normalization_173/mul_4/ReadVariableOp’/batch_instance_normalization_174/ReadVariableOp’1batch_instance_normalization_174/ReadVariableOp_1’5batch_instance_normalization_174/add_3/ReadVariableOp’5batch_instance_normalization_174/mul_4/ReadVariableOp’ conv2d_219/Conv2D/ReadVariableOp’ conv2d_220/Conv2D/ReadVariableOp’ conv2d_221/Conv2D/ReadVariableOp’ conv2d_222/Conv2D/ReadVariableOp’!conv2d_223/BiasAdd/ReadVariableOp’ conv2d_223/Conv2D/ReadVariableOpU
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  Cz
lambda/truedivRealDivinputs_1lambda/truediv/y:output:0*
T0*1
_output_shapes
:?????????Q
lambda/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x

lambda/subSublambda/sub/x:output:0lambda/truediv:z:0*
T0*1
_output_shapes
:?????????p
lambda/multiply/mulMulinputs_2lambda/sub:z:0*
T0*1
_output_shapes
:?????????v
lambda/multiply_1/mulMulinputs_0lambda/truediv:z:0*
T0*1
_output_shapes
:?????????
lambda/add/addAddV2lambda/multiply/mul:z:0lambda/multiply_1/mul:z:0*
T0*1
_output_shapes
:?????????
 conv2d_219/Conv2D/ReadVariableOpReadVariableOp)conv2d_219_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0»
conv2d_219/Conv2DConv2Dlambda/add/add:z:0(conv2d_219/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
c
	LeakyRelu	LeakyReluconv2d_219/Conv2D:output:0*/
_output_shapes
:?????????@@@
 conv2d_220/Conv2D/ReadVariableOpReadVariableOp)conv2d_220_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Α
conv2d_220/Conv2DConv2DLeakyRelu:activations:0(conv2d_220/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  *
paddingSAME*
strides

?batch_instance_normalization_172/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ή
-batch_instance_normalization_172/moments/meanMeanconv2d_220/Conv2D:output:0Hbatch_instance_normalization_172/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(―
5batch_instance_normalization_172/moments/StopGradientStopGradient6batch_instance_normalization_172/moments/mean:output:0*
T0*'
_output_shapes
:ζ
:batch_instance_normalization_172/moments/SquaredDifferenceSquaredDifferenceconv2d_220/Conv2D:output:0>batch_instance_normalization_172/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????  
Cbatch_instance_normalization_172/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_172/moments/varianceMean>batch_instance_normalization_172/moments/SquaredDifference:z:0Lbatch_instance_normalization_172/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ί
$batch_instance_normalization_172/subSubconv2d_220/Conv2D:output:06batch_instance_normalization_172/moments/mean:output:0*
T0*0
_output_shapes
:?????????  k
&batch_instance_normalization_172/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Μ
$batch_instance_normalization_172/addAddV2:batch_instance_normalization_172/moments/variance:output:0/batch_instance_normalization_172/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_172/RsqrtRsqrt(batch_instance_normalization_172/add:z:0*
T0*'
_output_shapes
:Ό
$batch_instance_normalization_172/mulMul(batch_instance_normalization_172/sub:z:0*batch_instance_normalization_172/Rsqrt:y:0*
T0*0
_output_shapes
:?????????  
Abatch_instance_normalization_172/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      λ
/batch_instance_normalization_172/moments_1/meanMeanconv2d_220/Conv2D:output:0Jbatch_instance_normalization_172/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ό
7batch_instance_normalization_172/moments_1/StopGradientStopGradient8batch_instance_normalization_172/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????κ
<batch_instance_normalization_172/moments_1/SquaredDifferenceSquaredDifferenceconv2d_220/Conv2D:output:0@batch_instance_normalization_172/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????  
Ebatch_instance_normalization_172/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_172/moments_1/varianceMean@batch_instance_normalization_172/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_172/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ύ
&batch_instance_normalization_172/sub_1Subconv2d_220/Conv2D:output:08batch_instance_normalization_172/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????  m
(batch_instance_normalization_172/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Ϋ
&batch_instance_normalization_172/add_1AddV2<batch_instance_normalization_172/moments_1/variance:output:01batch_instance_normalization_172/add_1/y:output:0*
T0*0
_output_shapes
:?????????
(batch_instance_normalization_172/Rsqrt_1Rsqrt*batch_instance_normalization_172/add_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_172/mul_1Mul*batch_instance_normalization_172/sub_1:z:0,batch_instance_normalization_172/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????  ₯
/batch_instance_normalization_172/ReadVariableOpReadVariableOp8batch_instance_normalization_172_readvariableop_resource*
_output_shapes	
:*
dtype0Λ
&batch_instance_normalization_172/mul_2Mul7batch_instance_normalization_172/ReadVariableOp:value:0(batch_instance_normalization_172/mul:z:0*
T0*0
_output_shapes
:?????????  §
1batch_instance_normalization_172/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_172_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_172/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Α
&batch_instance_normalization_172/sub_2Sub1batch_instance_normalization_172/sub_2/x:output:09batch_instance_normalization_172/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:ΐ
&batch_instance_normalization_172/mul_3Mul*batch_instance_normalization_172/sub_2:z:0*batch_instance_normalization_172/mul_1:z:0*
T0*0
_output_shapes
:?????????  Β
&batch_instance_normalization_172/add_2AddV2*batch_instance_normalization_172/mul_2:z:0*batch_instance_normalization_172/mul_3:z:0*
T0*0
_output_shapes
:?????????  ±
5batch_instance_normalization_172/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_172_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Σ
&batch_instance_normalization_172/mul_4Mul*batch_instance_normalization_172/add_2:z:0=batch_instance_normalization_172/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ±
5batch_instance_normalization_172/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_172_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Υ
&batch_instance_normalization_172/add_3AddV2*batch_instance_normalization_172/mul_4:z:0=batch_instance_normalization_172/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  v
LeakyRelu_1	LeakyRelu*batch_instance_normalization_172/add_3:z:0*0
_output_shapes
:?????????  
 conv2d_221/Conv2D/ReadVariableOpReadVariableOp)conv2d_221_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Γ
conv2d_221/Conv2DConv2DLeakyRelu_1:activations:0(conv2d_221/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

?batch_instance_normalization_173/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ή
-batch_instance_normalization_173/moments/meanMeanconv2d_221/Conv2D:output:0Hbatch_instance_normalization_173/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(―
5batch_instance_normalization_173/moments/StopGradientStopGradient6batch_instance_normalization_173/moments/mean:output:0*
T0*'
_output_shapes
:ζ
:batch_instance_normalization_173/moments/SquaredDifferenceSquaredDifferenceconv2d_221/Conv2D:output:0>batch_instance_normalization_173/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????
Cbatch_instance_normalization_173/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_173/moments/varianceMean>batch_instance_normalization_173/moments/SquaredDifference:z:0Lbatch_instance_normalization_173/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ί
$batch_instance_normalization_173/subSubconv2d_221/Conv2D:output:06batch_instance_normalization_173/moments/mean:output:0*
T0*0
_output_shapes
:?????????k
&batch_instance_normalization_173/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Μ
$batch_instance_normalization_173/addAddV2:batch_instance_normalization_173/moments/variance:output:0/batch_instance_normalization_173/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_173/RsqrtRsqrt(batch_instance_normalization_173/add:z:0*
T0*'
_output_shapes
:Ό
$batch_instance_normalization_173/mulMul(batch_instance_normalization_173/sub:z:0*batch_instance_normalization_173/Rsqrt:y:0*
T0*0
_output_shapes
:?????????
Abatch_instance_normalization_173/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      λ
/batch_instance_normalization_173/moments_1/meanMeanconv2d_221/Conv2D:output:0Jbatch_instance_normalization_173/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ό
7batch_instance_normalization_173/moments_1/StopGradientStopGradient8batch_instance_normalization_173/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????κ
<batch_instance_normalization_173/moments_1/SquaredDifferenceSquaredDifferenceconv2d_221/Conv2D:output:0@batch_instance_normalization_173/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????
Ebatch_instance_normalization_173/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_173/moments_1/varianceMean@batch_instance_normalization_173/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_173/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ύ
&batch_instance_normalization_173/sub_1Subconv2d_221/Conv2D:output:08batch_instance_normalization_173/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????m
(batch_instance_normalization_173/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Ϋ
&batch_instance_normalization_173/add_1AddV2<batch_instance_normalization_173/moments_1/variance:output:01batch_instance_normalization_173/add_1/y:output:0*
T0*0
_output_shapes
:?????????
(batch_instance_normalization_173/Rsqrt_1Rsqrt*batch_instance_normalization_173/add_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_173/mul_1Mul*batch_instance_normalization_173/sub_1:z:0,batch_instance_normalization_173/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????₯
/batch_instance_normalization_173/ReadVariableOpReadVariableOp8batch_instance_normalization_173_readvariableop_resource*
_output_shapes	
:*
dtype0Λ
&batch_instance_normalization_173/mul_2Mul7batch_instance_normalization_173/ReadVariableOp:value:0(batch_instance_normalization_173/mul:z:0*
T0*0
_output_shapes
:?????????§
1batch_instance_normalization_173/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_173_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_173/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Α
&batch_instance_normalization_173/sub_2Sub1batch_instance_normalization_173/sub_2/x:output:09batch_instance_normalization_173/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:ΐ
&batch_instance_normalization_173/mul_3Mul*batch_instance_normalization_173/sub_2:z:0*batch_instance_normalization_173/mul_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_173/add_2AddV2*batch_instance_normalization_173/mul_2:z:0*batch_instance_normalization_173/mul_3:z:0*
T0*0
_output_shapes
:?????????±
5batch_instance_normalization_173/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_173_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Σ
&batch_instance_normalization_173/mul_4Mul*batch_instance_normalization_173/add_2:z:0=batch_instance_normalization_173/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????±
5batch_instance_normalization_173/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_173_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Υ
&batch_instance_normalization_173/add_3AddV2*batch_instance_normalization_173/mul_4:z:0=batch_instance_normalization_173/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v
LeakyRelu_2	LeakyRelu*batch_instance_normalization_173/add_3:z:0*0
_output_shapes
:?????????
zero_padding2d_18/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
zero_padding2d_18/PadPadLeakyRelu_2:activations:0'zero_padding2d_18/Pad/paddings:output:0*
T0*0
_output_shapes
:?????????
 conv2d_222/Conv2D/ReadVariableOpReadVariableOp)conv2d_222_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ι
conv2d_222/Conv2DConv2Dzero_padding2d_18/Pad:output:0(conv2d_222/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides

?batch_instance_normalization_174/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ή
-batch_instance_normalization_174/moments/meanMeanconv2d_222/Conv2D:output:0Hbatch_instance_normalization_174/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(―
5batch_instance_normalization_174/moments/StopGradientStopGradient6batch_instance_normalization_174/moments/mean:output:0*
T0*'
_output_shapes
:ζ
:batch_instance_normalization_174/moments/SquaredDifferenceSquaredDifferenceconv2d_222/Conv2D:output:0>batch_instance_normalization_174/moments/StopGradient:output:0*
T0*0
_output_shapes
:?????????
Cbatch_instance_normalization_174/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_174/moments/varianceMean>batch_instance_normalization_174/moments/SquaredDifference:z:0Lbatch_instance_normalization_174/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ί
$batch_instance_normalization_174/subSubconv2d_222/Conv2D:output:06batch_instance_normalization_174/moments/mean:output:0*
T0*0
_output_shapes
:?????????k
&batch_instance_normalization_174/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Μ
$batch_instance_normalization_174/addAddV2:batch_instance_normalization_174/moments/variance:output:0/batch_instance_normalization_174/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_174/RsqrtRsqrt(batch_instance_normalization_174/add:z:0*
T0*'
_output_shapes
:Ό
$batch_instance_normalization_174/mulMul(batch_instance_normalization_174/sub:z:0*batch_instance_normalization_174/Rsqrt:y:0*
T0*0
_output_shapes
:?????????
Abatch_instance_normalization_174/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      λ
/batch_instance_normalization_174/moments_1/meanMeanconv2d_222/Conv2D:output:0Jbatch_instance_normalization_174/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ό
7batch_instance_normalization_174/moments_1/StopGradientStopGradient8batch_instance_normalization_174/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????κ
<batch_instance_normalization_174/moments_1/SquaredDifferenceSquaredDifferenceconv2d_222/Conv2D:output:0@batch_instance_normalization_174/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:?????????
Ebatch_instance_normalization_174/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_174/moments_1/varianceMean@batch_instance_normalization_174/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_174/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:?????????*
	keep_dims(Ύ
&batch_instance_normalization_174/sub_1Subconv2d_222/Conv2D:output:08batch_instance_normalization_174/moments_1/mean:output:0*
T0*0
_output_shapes
:?????????m
(batch_instance_normalization_174/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *¬Ε'7Ϋ
&batch_instance_normalization_174/add_1AddV2<batch_instance_normalization_174/moments_1/variance:output:01batch_instance_normalization_174/add_1/y:output:0*
T0*0
_output_shapes
:?????????
(batch_instance_normalization_174/Rsqrt_1Rsqrt*batch_instance_normalization_174/add_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_174/mul_1Mul*batch_instance_normalization_174/sub_1:z:0,batch_instance_normalization_174/Rsqrt_1:y:0*
T0*0
_output_shapes
:?????????₯
/batch_instance_normalization_174/ReadVariableOpReadVariableOp8batch_instance_normalization_174_readvariableop_resource*
_output_shapes	
:*
dtype0Λ
&batch_instance_normalization_174/mul_2Mul7batch_instance_normalization_174/ReadVariableOp:value:0(batch_instance_normalization_174/mul:z:0*
T0*0
_output_shapes
:?????????§
1batch_instance_normalization_174/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_174_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_174/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Α
&batch_instance_normalization_174/sub_2Sub1batch_instance_normalization_174/sub_2/x:output:09batch_instance_normalization_174/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:ΐ
&batch_instance_normalization_174/mul_3Mul*batch_instance_normalization_174/sub_2:z:0*batch_instance_normalization_174/mul_1:z:0*
T0*0
_output_shapes
:?????????Β
&batch_instance_normalization_174/add_2AddV2*batch_instance_normalization_174/mul_2:z:0*batch_instance_normalization_174/mul_3:z:0*
T0*0
_output_shapes
:?????????±
5batch_instance_normalization_174/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_174_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Σ
&batch_instance_normalization_174/mul_4Mul*batch_instance_normalization_174/add_2:z:0=batch_instance_normalization_174/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????±
5batch_instance_normalization_174/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_174_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Υ
&batch_instance_normalization_174/add_3AddV2*batch_instance_normalization_174/mul_4:z:0=batch_instance_normalization_174/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????v
LeakyRelu_3	LeakyRelu*batch_instance_normalization_174/add_3:z:0*0
_output_shapes
:?????????
zero_padding2d_19/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             
zero_padding2d_19/PadPadLeakyRelu_3:activations:0'zero_padding2d_19/Pad/paddings:output:0*
T0*0
_output_shapes
:?????????
 conv2d_223/Conv2D/ReadVariableOpReadVariableOp)conv2d_223_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype0Θ
conv2d_223/Conv2DConv2Dzero_padding2d_19/Pad:output:0(conv2d_223/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides

!conv2d_223/BiasAdd/ReadVariableOpReadVariableOp*conv2d_223_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_223/BiasAddBiasAddconv2d_223/Conv2D:output:0)conv2d_223/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????r
IdentityIdentityconv2d_223/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????
NoOpNoOp0^batch_instance_normalization_172/ReadVariableOp2^batch_instance_normalization_172/ReadVariableOp_16^batch_instance_normalization_172/add_3/ReadVariableOp6^batch_instance_normalization_172/mul_4/ReadVariableOp0^batch_instance_normalization_173/ReadVariableOp2^batch_instance_normalization_173/ReadVariableOp_16^batch_instance_normalization_173/add_3/ReadVariableOp6^batch_instance_normalization_173/mul_4/ReadVariableOp0^batch_instance_normalization_174/ReadVariableOp2^batch_instance_normalization_174/ReadVariableOp_16^batch_instance_normalization_174/add_3/ReadVariableOp6^batch_instance_normalization_174/mul_4/ReadVariableOp!^conv2d_219/Conv2D/ReadVariableOp!^conv2d_220/Conv2D/ReadVariableOp!^conv2d_221/Conv2D/ReadVariableOp!^conv2d_222/Conv2D/ReadVariableOp"^conv2d_223/BiasAdd/ReadVariableOp!^conv2d_223/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapesw
u:?????????:?????????:?????????: : : : : : : : : : : : : : : 2b
/batch_instance_normalization_172/ReadVariableOp/batch_instance_normalization_172/ReadVariableOp2f
1batch_instance_normalization_172/ReadVariableOp_11batch_instance_normalization_172/ReadVariableOp_12n
5batch_instance_normalization_172/add_3/ReadVariableOp5batch_instance_normalization_172/add_3/ReadVariableOp2n
5batch_instance_normalization_172/mul_4/ReadVariableOp5batch_instance_normalization_172/mul_4/ReadVariableOp2b
/batch_instance_normalization_173/ReadVariableOp/batch_instance_normalization_173/ReadVariableOp2f
1batch_instance_normalization_173/ReadVariableOp_11batch_instance_normalization_173/ReadVariableOp_12n
5batch_instance_normalization_173/add_3/ReadVariableOp5batch_instance_normalization_173/add_3/ReadVariableOp2n
5batch_instance_normalization_173/mul_4/ReadVariableOp5batch_instance_normalization_173/mul_4/ReadVariableOp2b
/batch_instance_normalization_174/ReadVariableOp/batch_instance_normalization_174/ReadVariableOp2f
1batch_instance_normalization_174/ReadVariableOp_11batch_instance_normalization_174/ReadVariableOp_12n
5batch_instance_normalization_174/add_3/ReadVariableOp5batch_instance_normalization_174/add_3/ReadVariableOp2n
5batch_instance_normalization_174/mul_4/ReadVariableOp5batch_instance_normalization_174/mul_4/ReadVariableOp2D
 conv2d_219/Conv2D/ReadVariableOp conv2d_219/Conv2D/ReadVariableOp2D
 conv2d_220/Conv2D/ReadVariableOp conv2d_220/Conv2D/ReadVariableOp2D
 conv2d_221/Conv2D/ReadVariableOp conv2d_221/Conv2D/ReadVariableOp2D
 conv2d_222/Conv2D/ReadVariableOp conv2d_222/Conv2D/ReadVariableOp2F
!conv2d_223/BiasAdd/ReadVariableOp!conv2d_223/BiasAdd/ReadVariableOp2D
 conv2d_223/Conv2D/ReadVariableOp conv2d_223/Conv2D/ReadVariableOp:[ W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/1:[W
1
_output_shapes
:?????????
"
_user_specified_name
inputs/2
«
»
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56549458

inputs:
conv2d_readvariableop_resource:
identity’Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????*
paddingVALID*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
 
_user_specified_nameinputs
υ?

$__inference__traced_restore_56549664
file_prefix<
"assignvariableop_conv2d_219_kernel:@?
$assignvariableop_1_conv2d_220_kernel:@F
7assignvariableop_2_batch_instance_normalization_172_rho:	H
9assignvariableop_3_batch_instance_normalization_172_gamma:	G
8assignvariableop_4_batch_instance_normalization_172_beta:	@
$assignvariableop_5_conv2d_221_kernel:F
7assignvariableop_6_batch_instance_normalization_173_rho:	H
9assignvariableop_7_batch_instance_normalization_173_gamma:	G
8assignvariableop_8_batch_instance_normalization_173_beta:	@
$assignvariableop_9_conv2d_222_kernel:G
8assignvariableop_10_batch_instance_normalization_174_rho:	I
:assignvariableop_11_batch_instance_normalization_174_gamma:	H
9assignvariableop_12_batch_instance_normalization_174_beta:	@
%assignvariableop_13_conv2d_223_kernel:1
#assignvariableop_14_conv2d_223_bias:
identity_16’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9μ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv4_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn4_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn4_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn4_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv5_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'conv5_1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B ξ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_219_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_220_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_2AssignVariableOp7assignvariableop_2_batch_instance_normalization_172_rhoIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_3AssignVariableOp9assignvariableop_3_batch_instance_normalization_172_gammaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_4AssignVariableOp8assignvariableop_4_batch_instance_normalization_172_betaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv2d_221_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_6AssignVariableOp7assignvariableop_6_batch_instance_normalization_173_rhoIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_instance_normalization_173_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_8AssignVariableOp8assignvariableop_8_batch_instance_normalization_173_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp$assignvariableop_9_conv2d_222_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:©
AssignVariableOp_10AssignVariableOp8assignvariableop_10_batch_instance_normalization_174_rhoIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_instance_normalization_174_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_12AssignVariableOp9assignvariableop_12_batch_instance_normalization_174_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp%assignvariableop_13_conv2d_223_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_223_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_16IdentityIdentity_15:output:0^NoOp_1*
T0*
_output_shapes
: 
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
_user_specified_namefile_prefix"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Λ
serving_default·
E
input_1:
serving_default_input_1:0?????????
E
input_2:
serving_default_input_2:0?????????
E
input_3:
serving_default_input_3:0?????????D
output_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:κ°
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
±

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
±

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
* &call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
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
±

*kernel
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
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
₯
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_layer
±

@kernel
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
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
₯
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
»

Vkernel
Wbias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses"
_tf_keras_layer

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

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
Κ
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
2
2__inference_face_d_region_4_layer_call_fn_56548522
2__inference_face_d_region_4_layer_call_fn_56548919
2__inference_face_d_region_4_layer_call_fn_56548956
2__inference_face_d_region_4_layer_call_fn_56548768΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
φ2σ
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56549103
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56549250
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548825
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548882΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΰBέ
#__inference__wrapped_model_56548234input_1input_2input_3"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
cserving_default"
signature_map
+:)@2conv2d_219/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
Χ2Τ
-__inference_conv2d_219_layer_call_fn_56549296’
²
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
annotationsͺ *
 
ς2ο
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56549303’
²
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
annotationsͺ *
 
,:*@2conv2d_220/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
Χ2Τ
-__inference_conv2d_220_layer_call_fn_56549310’
²
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
annotationsͺ *
 
ς2ο
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56549317’
²
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
annotationsͺ *
 
3:12$batch_instance_normalization_172/rho
5:32&batch_instance_normalization_172/gamma
4:22%batch_instance_normalization_172/beta
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
­
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
θ2ε
C__inference_batch_instance_normalization_172_layer_call_fn_56549328
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56549368
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
-:+2conv2d_221/kernel
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
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
Χ2Τ
-__inference_conv2d_221_layer_call_fn_56549375’
²
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
annotationsͺ *
 
ς2ο
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56549382’
²
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
annotationsͺ *
 
3:12$batch_instance_normalization_173/rho
5:32&batch_instance_normalization_173/gamma
4:22%batch_instance_normalization_173/beta
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
­
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
θ2ε
C__inference_batch_instance_normalization_173_layer_call_fn_56549393
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56549433
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
―
}non_trainable_variables

~layers
metrics
 layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
ή2Ϋ
4__inference_zero_padding2d_18_layer_call_fn_56549438’
²
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
annotationsͺ *
 
ω2φ
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56549444’
²
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
annotationsͺ *
 
-:+2conv2d_222/kernel
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
Χ2Τ
-__inference_conv2d_222_layer_call_fn_56549451’
²
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
annotationsͺ *
 
ς2ο
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56549458’
²
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
annotationsͺ *
 
3:12$batch_instance_normalization_174/rho
5:32&batch_instance_normalization_174/gamma
4:22%batch_instance_normalization_174/beta
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
θ2ε
C__inference_batch_instance_normalization_174_layer_call_fn_56549469
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56549509
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ή2Ϋ
4__inference_zero_padding2d_19_layer_call_fn_56549514’
²
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
annotationsͺ *
 
ω2φ
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56549520’
²
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
annotationsͺ *
 
,:*2conv2d_223/kernel
:2conv2d_223/bias
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
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
Χ2Τ
-__inference_conv2d_223_layer_call_fn_56549529’
²
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
annotationsͺ *
 
ς2ο
H__inference_conv2d_223_layer_call_and_return_conditional_losses_56549539’
²
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
annotationsͺ *
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
έBΪ
&__inference_signature_wrapper_56549289input_1input_2input_3"
²
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
annotationsͺ *
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
trackable_dict_wrapper
#__inference__wrapped_model_56548234ο!"#*123@GHIVW’
’

+(
input_1?????????
+(
input_2?????????
+(
input_3?????????
ͺ ";ͺ8
6
output_1*'
output_1?????????Μ
^__inference_batch_instance_normalization_172_layer_call_and_return_conditional_losses_56549368j!"#3’0
)’&
$!
x?????????  
ͺ ".’+
$!
0?????????  
 €
C__inference_batch_instance_normalization_172_layer_call_fn_56549328]!"#3’0
)’&
$!
x?????????  
ͺ "!?????????  Μ
^__inference_batch_instance_normalization_173_layer_call_and_return_conditional_losses_56549433j1233’0
)’&
$!
x?????????
ͺ ".’+
$!
0?????????
 €
C__inference_batch_instance_normalization_173_layer_call_fn_56549393]1233’0
)’&
$!
x?????????
ͺ "!?????????Μ
^__inference_batch_instance_normalization_174_layer_call_and_return_conditional_losses_56549509jGHI3’0
)’&
$!
x?????????
ͺ ".’+
$!
0?????????
 €
C__inference_batch_instance_normalization_174_layer_call_fn_56549469]GHI3’0
)’&
$!
x?????????
ͺ "!?????????Ή
H__inference_conv2d_219_layer_call_and_return_conditional_losses_56549303m9’6
/’,
*'
inputs?????????
ͺ "-’*
# 
0?????????@@@
 
-__inference_conv2d_219_layer_call_fn_56549296`9’6
/’,
*'
inputs?????????
ͺ " ?????????@@@Έ
H__inference_conv2d_220_layer_call_and_return_conditional_losses_56549317l7’4
-’*
(%
inputs?????????@@@
ͺ ".’+
$!
0?????????  
 
-__inference_conv2d_220_layer_call_fn_56549310_7’4
-’*
(%
inputs?????????@@@
ͺ "!?????????  Ή
H__inference_conv2d_221_layer_call_and_return_conditional_losses_56549382m*8’5
.’+
)&
inputs?????????  
ͺ ".’+
$!
0?????????
 
-__inference_conv2d_221_layer_call_fn_56549375`*8’5
.’+
)&
inputs?????????  
ͺ "!?????????Ή
H__inference_conv2d_222_layer_call_and_return_conditional_losses_56549458m@8’5
.’+
)&
inputs?????????
ͺ ".’+
$!
0?????????
 
-__inference_conv2d_222_layer_call_fn_56549451`@8’5
.’+
)&
inputs?????????
ͺ "!?????????Ή
H__inference_conv2d_223_layer_call_and_return_conditional_losses_56549539mVW8’5
.’+
)&
inputs?????????
ͺ "-’*
# 
0?????????
 
-__inference_conv2d_223_layer_call_fn_56549529`VW8’5
.’+
)&
inputs?????????
ͺ " ?????????·
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548825ε!"#*123@GHIVW’’
’

+(
input_1?????????
+(
input_2?????????
+(
input_3?????????
p 
ͺ "-’*
# 
0?????????
 ·
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56548882ε!"#*123@GHIVW’’
’

+(
input_1?????????
+(
input_2?????????
+(
input_3?????????
p
ͺ "-’*
# 
0?????????
 Ί
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56549103θ!"#*123@GHIVW₯’‘
’

,)
inputs/0?????????
,)
inputs/1?????????
,)
inputs/2?????????
p 
ͺ "-’*
# 
0?????????
 Ί
M__inference_face_d_region_4_layer_call_and_return_conditional_losses_56549250θ!"#*123@GHIVW₯’‘
’

,)
inputs/0?????????
,)
inputs/1?????????
,)
inputs/2?????????
p
ͺ "-’*
# 
0?????????
 
2__inference_face_d_region_4_layer_call_fn_56548522Ψ!"#*123@GHIVW’’
’

+(
input_1?????????
+(
input_2?????????
+(
input_3?????????
p 
ͺ " ?????????
2__inference_face_d_region_4_layer_call_fn_56548768Ψ!"#*123@GHIVW’’
’

+(
input_1?????????
+(
input_2?????????
+(
input_3?????????
p
ͺ " ?????????
2__inference_face_d_region_4_layer_call_fn_56548919Ϋ!"#*123@GHIVW₯’‘
’

,)
inputs/0?????????
,)
inputs/1?????????
,)
inputs/2?????????
p 
ͺ " ?????????
2__inference_face_d_region_4_layer_call_fn_56548956Ϋ!"#*123@GHIVW₯’‘
’

,)
inputs/0?????????
,)
inputs/1?????????
,)
inputs/2?????????
p
ͺ " ?????????΄
&__inference_signature_wrapper_56549289!"#*123@GHIVWΈ’΄
’ 
¬ͺ¨
6
input_1+(
input_1?????????
6
input_2+(
input_2?????????
6
input_3+(
input_3?????????";ͺ8
6
output_1*'
output_1?????????ς
O__inference_zero_padding2d_18_layer_call_and_return_conditional_losses_56549444R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Κ
4__inference_zero_padding2d_18_layer_call_fn_56549438R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????ς
O__inference_zero_padding2d_19_layer_call_and_return_conditional_losses_56549520R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Κ
4__inference_zero_padding2d_19_layer_call_fn_56549514R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????