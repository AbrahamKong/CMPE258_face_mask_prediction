Ķī/
á
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
 
BatchToSpaceND

input"T
block_shape"Tblock_shape
crops"Tcrops
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
Tcropstype0:
2	
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
Ā
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
alphafloat%ÍĖL>"
Ttype0:
2

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
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
Đ
SpaceToBatchND

input"T
block_shape"Tblock_shape
paddings"	Tpaddings
output"T"	
Ttype" 
Tblock_shapetype0:
2	"
	Tpaddingstype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
Á
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
executor_typestring Ļ
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
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-0-g3f878cff5b68Ũ+

conv2d_606/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_606/kernel

%conv2d_606/kernel/Read/ReadVariableOpReadVariableOpconv2d_606/kernel*&
_output_shapes
:@*
dtype0

conv2d_607/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*"
shared_nameconv2d_607/kernel

%conv2d_607/kernel/Read/ReadVariableOpReadVariableOpconv2d_607/kernel*&
_output_shapes
:@@*
dtype0

conv2d_608/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_608/kernel

%conv2d_608/kernel/Read/ReadVariableOpReadVariableOpconv2d_608/kernel*'
_output_shapes
:@*
dtype0
Ą
$batch_instance_normalization_500/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_500/rho

8batch_instance_normalization_500/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_500/rho*
_output_shapes	
:*
dtype0
Ĩ
&batch_instance_normalization_500/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_500/gamma

:batch_instance_normalization_500/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_500/gamma*
_output_shapes	
:*
dtype0
Ģ
%batch_instance_normalization_500/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_500/beta

9batch_instance_normalization_500/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_500/beta*
_output_shapes	
:*
dtype0

conv2d_609/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_609/kernel

%conv2d_609/kernel/Read/ReadVariableOpReadVariableOpconv2d_609/kernel*(
_output_shapes
:*
dtype0
Ą
$batch_instance_normalization_501/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_501/rho

8batch_instance_normalization_501/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_501/rho*
_output_shapes	
:*
dtype0
Ĩ
&batch_instance_normalization_501/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_501/gamma

:batch_instance_normalization_501/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_501/gamma*
_output_shapes	
:*
dtype0
Ģ
%batch_instance_normalization_501/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_501/beta

9batch_instance_normalization_501/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_501/beta*
_output_shapes	
:*
dtype0

conv2d_610/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_610/kernel

%conv2d_610/kernel/Read/ReadVariableOpReadVariableOpconv2d_610/kernel*(
_output_shapes
:*
dtype0
Ą
$batch_instance_normalization_502/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_502/rho

8batch_instance_normalization_502/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_502/rho*
_output_shapes	
:*
dtype0
Ĩ
&batch_instance_normalization_502/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_502/gamma

:batch_instance_normalization_502/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_502/gamma*
_output_shapes	
:*
dtype0
Ģ
%batch_instance_normalization_502/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_502/beta

9batch_instance_normalization_502/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_502/beta*
_output_shapes	
:*
dtype0

conv2d_611/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_611/kernel

%conv2d_611/kernel/Read/ReadVariableOpReadVariableOpconv2d_611/kernel*(
_output_shapes
:*
dtype0
Ą
$batch_instance_normalization_503/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_503/rho

8batch_instance_normalization_503/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_503/rho*
_output_shapes	
:*
dtype0
Ĩ
&batch_instance_normalization_503/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_503/gamma

:batch_instance_normalization_503/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_503/gamma*
_output_shapes	
:*
dtype0
Ģ
%batch_instance_normalization_503/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_503/beta

9batch_instance_normalization_503/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_503/beta*
_output_shapes	
:*
dtype0

conv2d_612/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_612/kernel

%conv2d_612/kernel/Read/ReadVariableOpReadVariableOpconv2d_612/kernel*(
_output_shapes
:*
dtype0
Ą
$batch_instance_normalization_504/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_504/rho

8batch_instance_normalization_504/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_504/rho*
_output_shapes	
:*
dtype0
Ĩ
&batch_instance_normalization_504/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_504/gamma

:batch_instance_normalization_504/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_504/gamma*
_output_shapes	
:*
dtype0
Ģ
%batch_instance_normalization_504/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_504/beta

9batch_instance_normalization_504/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_504/beta*
_output_shapes	
:*
dtype0

conv2d_613/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_613/kernel

%conv2d_613/kernel/Read/ReadVariableOpReadVariableOpconv2d_613/kernel*(
_output_shapes
:*
dtype0
Ą
$batch_instance_normalization_505/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_505/rho

8batch_instance_normalization_505/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_505/rho*
_output_shapes	
:*
dtype0
Ĩ
&batch_instance_normalization_505/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_505/gamma

:batch_instance_normalization_505/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_505/gamma*
_output_shapes	
:*
dtype0
Ģ
%batch_instance_normalization_505/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_505/beta

9batch_instance_normalization_505/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_505/beta*
_output_shapes	
:*
dtype0

conv2d_transpose_100/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameconv2d_transpose_100/kernel

/conv2d_transpose_100/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_100/kernel*(
_output_shapes
:*
dtype0

conv2d_614/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_614/kernel

%conv2d_614/kernel/Read/ReadVariableOpReadVariableOpconv2d_614/kernel*(
_output_shapes
:*
dtype0
Ą
$batch_instance_normalization_506/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_506/rho

8batch_instance_normalization_506/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_506/rho*
_output_shapes	
:*
dtype0
Ĩ
&batch_instance_normalization_506/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_506/gamma

:batch_instance_normalization_506/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_506/gamma*
_output_shapes	
:*
dtype0
Ģ
%batch_instance_normalization_506/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_506/beta

9batch_instance_normalization_506/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_506/beta*
_output_shapes	
:*
dtype0

conv2d_615/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_615/kernel

%conv2d_615/kernel/Read/ReadVariableOpReadVariableOpconv2d_615/kernel*(
_output_shapes
:*
dtype0
Ą
$batch_instance_normalization_507/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_507/rho

8batch_instance_normalization_507/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_507/rho*
_output_shapes	
:*
dtype0
Ĩ
&batch_instance_normalization_507/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_507/gamma

:batch_instance_normalization_507/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_507/gamma*
_output_shapes	
:*
dtype0
Ģ
%batch_instance_normalization_507/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_507/beta

9batch_instance_normalization_507/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_507/beta*
_output_shapes	
:*
dtype0

conv2d_transpose_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_nameconv2d_transpose_101/kernel

/conv2d_transpose_101/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_101/kernel*'
_output_shapes
:@*
dtype0

conv2d_616/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_616/kernel

%conv2d_616/kernel/Read/ReadVariableOpReadVariableOpconv2d_616/kernel*'
_output_shapes
:@*
dtype0
 
$batch_instance_normalization_508/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$batch_instance_normalization_508/rho

8batch_instance_normalization_508/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_508/rho*
_output_shapes
:@*
dtype0
Ī
&batch_instance_normalization_508/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_instance_normalization_508/gamma

:batch_instance_normalization_508/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_508/gamma*
_output_shapes
:@*
dtype0
Ē
%batch_instance_normalization_508/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_instance_normalization_508/beta

9batch_instance_normalization_508/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_508/beta*
_output_shapes
:@*
dtype0

conv2d_617/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*"
shared_nameconv2d_617/kernel

%conv2d_617/kernel/Read/ReadVariableOpReadVariableOpconv2d_617/kernel*&
_output_shapes
:@@*
dtype0
 
$batch_instance_normalization_509/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$batch_instance_normalization_509/rho

8batch_instance_normalization_509/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_509/rho*
_output_shapes
:@*
dtype0
Ī
&batch_instance_normalization_509/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_instance_normalization_509/gamma

:batch_instance_normalization_509/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_509/gamma*
_output_shapes
:@*
dtype0
Ē
%batch_instance_normalization_509/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_instance_normalization_509/beta

9batch_instance_normalization_509/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_509/beta*
_output_shapes
:@*
dtype0

conv2d_618/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_618/kernel

%conv2d_618/kernel/Read/ReadVariableOpReadVariableOpconv2d_618/kernel*&
_output_shapes
:@*
dtype0
 
$batch_instance_normalization_510/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$batch_instance_normalization_510/rho

8batch_instance_normalization_510/rho/Read/ReadVariableOpReadVariableOp$batch_instance_normalization_510/rho*
_output_shapes
:*
dtype0
Ī
&batch_instance_normalization_510/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_instance_normalization_510/gamma

:batch_instance_normalization_510/gamma/Read/ReadVariableOpReadVariableOp&batch_instance_normalization_510/gamma*
_output_shapes
:*
dtype0
Ē
%batch_instance_normalization_510/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_instance_normalization_510/beta

9batch_instance_normalization_510/beta/Read/ReadVariableOpReadVariableOp%batch_instance_normalization_510/beta*
_output_shapes
:*
dtype0

conv2d_619/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_619/kernel

%conv2d_619/kernel/Read/ReadVariableOpReadVariableOpconv2d_619/kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
°Ž
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ęŦ
valueßŦBÛŦ BÓŦ

conv1_1
conv1_2
	pool1
conv2_1
	bn2_1
conv2_2
	bn2_2
	pool2
	conv3_1
	
bn3_1
conv3_2
	bn3_2
conv3_3
	bn3_3
conv3_4
	bn3_4

convt1
conv6_1
	bn6_1
conv6_2
	bn6_2

convt2
conv7_1
	bn7_1
conv7_2
	bn7_2
conv7_3
	bn7_3
out
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%
signatures*


&kernel
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses*


-kernel
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses*

4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses* 


:kernel
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses*
Ū
Arho
	Bgamma
Cbeta
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*


Jkernel
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses*
Ū
Qrho
	Rgamma
Sbeta
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*

Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses* 


`kernel
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses*
Ū
grho
	hgamma
ibeta
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses*


pkernel
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses*
Ū
wrho
	xgamma
ybeta
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses*
Ģ
kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
·
rho

gamma
	beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ģ
kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
·
rho

gamma
	beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ģ
 kernel
Ą	variables
Ētrainable_variables
Ģregularization_losses
Ī	keras_api
Ĩ__call__
+Ķ&call_and_return_all_conditional_losses*
Ģ
§kernel
Ļ	variables
Đtrainable_variables
Šregularization_losses
Ŧ	keras_api
Ž__call__
+­&call_and_return_all_conditional_losses*
·
Ūrho

Ŋgamma
	°beta
ą	variables
ētrainable_variables
ģregularization_losses
ī	keras_api
ĩ__call__
+ķ&call_and_return_all_conditional_losses*
Ģ
·kernel
ļ	variables
đtrainable_variables
šregularization_losses
ŧ	keras_api
ž__call__
+―&call_and_return_all_conditional_losses*
·
ūrho

ŋgamma
	Ābeta
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses*
Ģ
Įkernel
Č	variables
Étrainable_variables
Ęregularization_losses
Ë	keras_api
Ė__call__
+Í&call_and_return_all_conditional_losses*
Ģ
Îkernel
Ï	variables
Ðtrainable_variables
Ņregularization_losses
Ō	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses*
·
Õrho

Ögamma
	Ũbeta
Ø	variables
Ųtrainable_variables
Úregularization_losses
Û	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses*
Ģ
Þkernel
ß	variables
ātrainable_variables
áregularization_losses
â	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses*
·
årho

ægamma
	įbeta
č	variables
étrainable_variables
ęregularization_losses
ë	keras_api
ė__call__
+í&call_and_return_all_conditional_losses*
Ģ
îkernel
ï	variables
ðtrainable_variables
ņregularization_losses
ō	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses*
·
õrho

ögamma
	ũbeta
ø	variables
ųtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses*
Ģ
þkernel
ĸ	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*
Ą
&0
-1
:2
A3
B4
C5
J6
Q7
R8
S9
`10
g11
h12
i13
p14
w15
x16
y17
18
19
20
21
22
23
24
25
 26
§27
Ū28
Ŋ29
°30
·31
ū32
ŋ33
Ā34
Į35
Î36
Õ37
Ö38
Ũ39
Þ40
å41
æ42
į43
î44
õ45
ö46
ũ47
þ48*
Ą
&0
-1
:2
A3
B4
C5
J6
Q7
R8
S9
`10
g11
h12
i13
p14
w15
x16
y17
18
19
20
21
22
23
24
25
 26
§27
Ū28
Ŋ29
°30
·31
ū32
ŋ33
Ā34
Į35
Î36
Õ37
Ö38
Ũ39
Þ40
å41
æ42
į43
î44
õ45
ö46
ũ47
þ48*
* 
ĩ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 

serving_default* 
TN
VARIABLE_VALUEconv2d_606/kernel)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

&0*

&0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_607/kernel)conv1_2/kernel/.ATTRIBUTES/VARIABLE_VALUE*

-0*

-0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses* 
* 
* 
TN
VARIABLE_VALUEconv2d_608/kernel)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

:0*

:0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_500/rho$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_500/gamma&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_500/beta%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1
C2*

A0
B1
C2*
* 

non_trainable_variables
 layers
Ąmetrics
 Ēlayer_regularization_losses
Ģlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_609/kernel)conv2_2/kernel/.ATTRIBUTES/VARIABLE_VALUE*

J0*

J0*
* 

Īnon_trainable_variables
Ĩlayers
Ķmetrics
 §layer_regularization_losses
Ļlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_501/rho$bn2_2/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_501/gamma&bn2_2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_501/beta%bn2_2/beta/.ATTRIBUTES/VARIABLE_VALUE*

Q0
R1
S2*

Q0
R1
S2*
* 

Đnon_trainable_variables
Šlayers
Ŧmetrics
 Žlayer_regularization_losses
­layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

Ūnon_trainable_variables
Ŋlayers
°metrics
 ąlayer_regularization_losses
ēlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses* 
* 
* 
TN
VARIABLE_VALUEconv2d_610/kernel)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

`0*

`0*
* 

ģnon_trainable_variables
īlayers
ĩmetrics
 ķlayer_regularization_losses
·layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_502/rho$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_502/gamma&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_502/beta%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

g0
h1
i2*

g0
h1
i2*
* 

ļnon_trainable_variables
đlayers
šmetrics
 ŧlayer_regularization_losses
žlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_611/kernel)conv3_2/kernel/.ATTRIBUTES/VARIABLE_VALUE*

p0*

p0*
* 

―non_trainable_variables
ūlayers
ŋmetrics
 Ālayer_regularization_losses
Álayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_503/rho$bn3_2/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_503/gamma&bn3_2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_503/beta%bn3_2/beta/.ATTRIBUTES/VARIABLE_VALUE*

w0
x1
y2*

w0
x1
y2*
* 

Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_612/kernel)conv3_3/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

Įnon_trainable_variables
Člayers
Émetrics
 Ęlayer_regularization_losses
Ëlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_504/rho$bn3_3/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_504/gamma&bn3_3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_504/beta%bn3_3/beta/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*
* 

Ėnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_613/kernel)conv3_4/kernel/.ATTRIBUTES/VARIABLE_VALUE*

0*

0*
* 

Ņnon_trainable_variables
Ōlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_505/rho$bn3_4/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_505/gamma&bn3_4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_505/beta%bn3_4/beta/.ATTRIBUTES/VARIABLE_VALUE*

0
1
2*

0
1
2*
* 

Önon_trainable_variables
Ũlayers
Ømetrics
 Ųlayer_regularization_losses
Úlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEconv2d_transpose_100/kernel(convt1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

 0*

 0*
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
Ą	variables
Ētrainable_variables
Ģregularization_losses
Ĩ__call__
+Ķ&call_and_return_all_conditional_losses
'Ķ"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_614/kernel)conv6_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

§0*

§0*
* 

ānon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
Ļ	variables
Đtrainable_variables
Šregularization_losses
Ž__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_506/rho$bn6_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_506/gamma&bn6_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_506/beta%bn6_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

Ū0
Ŋ1
°2*

Ū0
Ŋ1
°2*
* 

ånon_trainable_variables
ælayers
įmetrics
 člayer_regularization_losses
élayer_metrics
ą	variables
ētrainable_variables
ģregularization_losses
ĩ__call__
+ķ&call_and_return_all_conditional_losses
'ķ"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_615/kernel)conv6_2/kernel/.ATTRIBUTES/VARIABLE_VALUE*

·0*

·0*
* 

ęnon_trainable_variables
ëlayers
ėmetrics
 ílayer_regularization_losses
îlayer_metrics
ļ	variables
đtrainable_variables
šregularization_losses
ž__call__
+―&call_and_return_all_conditional_losses
'―"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_507/rho$bn6_2/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_507/gamma&bn6_2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_507/beta%bn6_2/beta/.ATTRIBUTES/VARIABLE_VALUE*

ū0
ŋ1
Ā2*

ū0
ŋ1
Ā2*
* 

ïnon_trainable_variables
ðlayers
ņmetrics
 ōlayer_regularization_losses
ólayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses*
* 
* 
]W
VARIABLE_VALUEconv2d_transpose_101/kernel(convt2/kernel/.ATTRIBUTES/VARIABLE_VALUE*

Į0*

Į0*
* 

ônon_trainable_variables
õlayers
ömetrics
 ũlayer_regularization_losses
ølayer_metrics
Č	variables
Étrainable_variables
Ęregularization_losses
Ė__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_616/kernel)conv7_1/kernel/.ATTRIBUTES/VARIABLE_VALUE*

Î0*

Î0*
* 

ųnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
Ï	variables
Ðtrainable_variables
Ņregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_508/rho$bn7_1/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_508/gamma&bn7_1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_508/beta%bn7_1/beta/.ATTRIBUTES/VARIABLE_VALUE*

Õ0
Ö1
Ũ2*

Õ0
Ö1
Ũ2*
* 

þnon_trainable_variables
ĸlayers
metrics
 layer_regularization_losses
layer_metrics
Ø	variables
Ųtrainable_variables
Úregularization_losses
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_617/kernel)conv7_2/kernel/.ATTRIBUTES/VARIABLE_VALUE*

Þ0*

Þ0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ß	variables
ātrainable_variables
áregularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_509/rho$bn7_2/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_509/gamma&bn7_2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_509/beta%bn7_2/beta/.ATTRIBUTES/VARIABLE_VALUE*

å0
æ1
į2*

å0
æ1
į2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
č	variables
étrainable_variables
ęregularization_losses
ė__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses*
* 
* 
TN
VARIABLE_VALUEconv2d_618/kernel)conv7_3/kernel/.ATTRIBUTES/VARIABLE_VALUE*

î0*

î0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ï	variables
ðtrainable_variables
ņregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses*
* 
* 
b\
VARIABLE_VALUE$batch_instance_normalization_510/rho$bn7_3/rho/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE&batch_instance_normalization_510/gamma&bn7_3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE%batch_instance_normalization_510/beta%bn7_3/beta/.ATTRIBUTES/VARIABLE_VALUE*

õ0
ö1
ũ2*

õ0
ö1
ũ2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ø	variables
ųtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses*
* 
* 
PJ
VARIABLE_VALUEconv2d_619/kernel%out/kernel/.ATTRIBUTES/VARIABLE_VALUE*

þ0*

þ0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ĸ	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
* 
* 
* 
â
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
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28*
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
:ĸĸĸĸĸĸĸĸĸ*
dtype0*&
shape:ĸĸĸĸĸĸĸĸĸ

serving_default_input_2Placeholder*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
dtype0*&
shape:ĸĸĸĸĸĸĸĸĸ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2conv2d_606/kernelconv2d_607/kernelconv2d_608/kernel$batch_instance_normalization_500/rho&batch_instance_normalization_500/gamma%batch_instance_normalization_500/betaconv2d_609/kernel$batch_instance_normalization_501/rho&batch_instance_normalization_501/gamma%batch_instance_normalization_501/betaconv2d_610/kernel$batch_instance_normalization_502/rho&batch_instance_normalization_502/gamma%batch_instance_normalization_502/betaconv2d_611/kernel$batch_instance_normalization_503/rho&batch_instance_normalization_503/gamma%batch_instance_normalization_503/betaconv2d_612/kernel$batch_instance_normalization_504/rho&batch_instance_normalization_504/gamma%batch_instance_normalization_504/betaconv2d_613/kernel$batch_instance_normalization_505/rho&batch_instance_normalization_505/gamma%batch_instance_normalization_505/betaconv2d_transpose_100/kernelconv2d_614/kernel$batch_instance_normalization_506/rho&batch_instance_normalization_506/gamma%batch_instance_normalization_506/betaconv2d_615/kernel$batch_instance_normalization_507/rho&batch_instance_normalization_507/gamma%batch_instance_normalization_507/betaconv2d_transpose_101/kernelconv2d_616/kernel$batch_instance_normalization_508/rho&batch_instance_normalization_508/gamma%batch_instance_normalization_508/betaconv2d_617/kernel$batch_instance_normalization_509/rho&batch_instance_normalization_509/gamma%batch_instance_normalization_509/betaconv2d_618/kernel$batch_instance_normalization_510/rho&batch_instance_normalization_510/gamma%batch_instance_normalization_510/betaconv2d_619/kernel*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_56545074
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ð
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_606/kernel/Read/ReadVariableOp%conv2d_607/kernel/Read/ReadVariableOp%conv2d_608/kernel/Read/ReadVariableOp8batch_instance_normalization_500/rho/Read/ReadVariableOp:batch_instance_normalization_500/gamma/Read/ReadVariableOp9batch_instance_normalization_500/beta/Read/ReadVariableOp%conv2d_609/kernel/Read/ReadVariableOp8batch_instance_normalization_501/rho/Read/ReadVariableOp:batch_instance_normalization_501/gamma/Read/ReadVariableOp9batch_instance_normalization_501/beta/Read/ReadVariableOp%conv2d_610/kernel/Read/ReadVariableOp8batch_instance_normalization_502/rho/Read/ReadVariableOp:batch_instance_normalization_502/gamma/Read/ReadVariableOp9batch_instance_normalization_502/beta/Read/ReadVariableOp%conv2d_611/kernel/Read/ReadVariableOp8batch_instance_normalization_503/rho/Read/ReadVariableOp:batch_instance_normalization_503/gamma/Read/ReadVariableOp9batch_instance_normalization_503/beta/Read/ReadVariableOp%conv2d_612/kernel/Read/ReadVariableOp8batch_instance_normalization_504/rho/Read/ReadVariableOp:batch_instance_normalization_504/gamma/Read/ReadVariableOp9batch_instance_normalization_504/beta/Read/ReadVariableOp%conv2d_613/kernel/Read/ReadVariableOp8batch_instance_normalization_505/rho/Read/ReadVariableOp:batch_instance_normalization_505/gamma/Read/ReadVariableOp9batch_instance_normalization_505/beta/Read/ReadVariableOp/conv2d_transpose_100/kernel/Read/ReadVariableOp%conv2d_614/kernel/Read/ReadVariableOp8batch_instance_normalization_506/rho/Read/ReadVariableOp:batch_instance_normalization_506/gamma/Read/ReadVariableOp9batch_instance_normalization_506/beta/Read/ReadVariableOp%conv2d_615/kernel/Read/ReadVariableOp8batch_instance_normalization_507/rho/Read/ReadVariableOp:batch_instance_normalization_507/gamma/Read/ReadVariableOp9batch_instance_normalization_507/beta/Read/ReadVariableOp/conv2d_transpose_101/kernel/Read/ReadVariableOp%conv2d_616/kernel/Read/ReadVariableOp8batch_instance_normalization_508/rho/Read/ReadVariableOp:batch_instance_normalization_508/gamma/Read/ReadVariableOp9batch_instance_normalization_508/beta/Read/ReadVariableOp%conv2d_617/kernel/Read/ReadVariableOp8batch_instance_normalization_509/rho/Read/ReadVariableOp:batch_instance_normalization_509/gamma/Read/ReadVariableOp9batch_instance_normalization_509/beta/Read/ReadVariableOp%conv2d_618/kernel/Read/ReadVariableOp8batch_instance_normalization_510/rho/Read/ReadVariableOp:batch_instance_normalization_510/gamma/Read/ReadVariableOp9batch_instance_normalization_510/beta/Read/ReadVariableOp%conv2d_619/kernel/Read/ReadVariableOpConst*>
Tin7
523*
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
!__inference__traced_save_56546144

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_606/kernelconv2d_607/kernelconv2d_608/kernel$batch_instance_normalization_500/rho&batch_instance_normalization_500/gamma%batch_instance_normalization_500/betaconv2d_609/kernel$batch_instance_normalization_501/rho&batch_instance_normalization_501/gamma%batch_instance_normalization_501/betaconv2d_610/kernel$batch_instance_normalization_502/rho&batch_instance_normalization_502/gamma%batch_instance_normalization_502/betaconv2d_611/kernel$batch_instance_normalization_503/rho&batch_instance_normalization_503/gamma%batch_instance_normalization_503/betaconv2d_612/kernel$batch_instance_normalization_504/rho&batch_instance_normalization_504/gamma%batch_instance_normalization_504/betaconv2d_613/kernel$batch_instance_normalization_505/rho&batch_instance_normalization_505/gamma%batch_instance_normalization_505/betaconv2d_transpose_100/kernelconv2d_614/kernel$batch_instance_normalization_506/rho&batch_instance_normalization_506/gamma%batch_instance_normalization_506/betaconv2d_615/kernel$batch_instance_normalization_507/rho&batch_instance_normalization_507/gamma%batch_instance_normalization_507/betaconv2d_transpose_101/kernelconv2d_616/kernel$batch_instance_normalization_508/rho&batch_instance_normalization_508/gamma%batch_instance_normalization_508/betaconv2d_617/kernel$batch_instance_normalization_509/rho&batch_instance_normalization_509/gamma%batch_instance_normalization_509/betaconv2d_618/kernel$batch_instance_normalization_510/rho&batch_instance_normalization_510/gamma%batch_instance_normalization_510/betaconv2d_619/kernel*=
Tin6
422*
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
$__inference__traced_restore_56546301ūû(
Š
ŧ
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56541897

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
Õ

-__inference_conv2d_618_layer_call_fn_56545901

inputs!
unknown:@
identityĒStatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56542496y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
ķ
Ā
C__inference_batch_instance_normalization_509_layer_call_fn_56545854
x
unknown:@
	unknown_0:@
	unknown_1:@
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56542480y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@

_user_specified_namex
Ó

-__inference_conv2d_614_layer_call_fn_56545604

inputs#
unknown:
identityĒStatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56542251x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56545483
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ  w
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
:ĸĸĸĸĸĸĸĸĸ  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  e
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
:ĸĸĸĸĸĸĸĸĸ  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
·$
Î
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56545727
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ@@w
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
:ĸĸĸĸĸĸĸĸĸ@@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ@@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@e
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
:ĸĸĸĸĸĸĸĸĸ@@_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
Š
ŧ
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56545676

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56542230
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ  w
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
:ĸĸĸĸĸĸĸĸĸ  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  e
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
:ĸĸĸĸĸĸĸĸĸ  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
Â
Ø
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56541790

inputsC
(conv2d_transpose_readvariableop_resource:@
identityĒconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ņ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*A
_output_shapes/
-:+ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56542086
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ  w
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
:ĸĸĸĸĸĸĸĸĸ  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  e
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
:ĸĸĸĸĸĸĸĸĸ  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
Ð

-__inference_conv2d_608_layer_call_fn_56545119

inputs"
unknown:@
identityĒStatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56541837x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ĸĸĸĸĸĸĸĸĸ@@@: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56542158
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ  w
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
:ĸĸĸĸĸĸĸĸĸ  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  e
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
:ĸĸĸĸĸĸĸĸĸ  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
ģ
ŧ
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56542042

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOpe
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      l
Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            m
Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                r
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ŋ
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ

~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ģ
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
r
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      |
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                Å
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
IdentityIdentityConv2D/BatchToSpaceND:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56541941
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ@@w
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
:ĸĸĸĸĸĸĸĸĸ@@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ@@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@e
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
:ĸĸĸĸĸĸĸĸĸ@@_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
·$
Î
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56542014
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ  w
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
:ĸĸĸĸĸĸĸĸĸ  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  e
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
:ĸĸĸĸĸĸĸĸĸ  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
·$
Î
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56542295
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ@@w
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
:ĸĸĸĸĸĸĸĸĸ@@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ@@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@e
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
:ĸĸĸĸĸĸĸĸĸ@@_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
íŋ
é
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543521
input_1
input_2-
conv2d_606_56543376:@-
conv2d_607_56543379:@@.
conv2d_608_56543384:@8
)batch_instance_normalization_500_56543387:	8
)batch_instance_normalization_500_56543389:	8
)batch_instance_normalization_500_56543391:	/
conv2d_609_56543395:8
)batch_instance_normalization_501_56543398:	8
)batch_instance_normalization_501_56543400:	8
)batch_instance_normalization_501_56543402:	/
conv2d_610_56543407:8
)batch_instance_normalization_502_56543410:	8
)batch_instance_normalization_502_56543412:	8
)batch_instance_normalization_502_56543414:	/
conv2d_611_56543418:8
)batch_instance_normalization_503_56543421:	8
)batch_instance_normalization_503_56543423:	8
)batch_instance_normalization_503_56543425:	/
conv2d_612_56543429:8
)batch_instance_normalization_504_56543432:	8
)batch_instance_normalization_504_56543434:	8
)batch_instance_normalization_504_56543436:	/
conv2d_613_56543440:8
)batch_instance_normalization_505_56543443:	8
)batch_instance_normalization_505_56543445:	8
)batch_instance_normalization_505_56543447:	9
conv2d_transpose_100_56543451:/
conv2d_614_56543456:8
)batch_instance_normalization_506_56543459:	8
)batch_instance_normalization_506_56543461:	8
)batch_instance_normalization_506_56543463:	/
conv2d_615_56543467:8
)batch_instance_normalization_507_56543470:	8
)batch_instance_normalization_507_56543472:	8
)batch_instance_normalization_507_56543474:	8
conv2d_transpose_101_56543478:@.
conv2d_616_56543483:@7
)batch_instance_normalization_508_56543486:@7
)batch_instance_normalization_508_56543488:@7
)batch_instance_normalization_508_56543490:@-
conv2d_617_56543494:@@7
)batch_instance_normalization_509_56543497:@7
)batch_instance_normalization_509_56543499:@7
)batch_instance_normalization_509_56543501:@-
conv2d_618_56543505:@7
)batch_instance_normalization_510_56543508:7
)batch_instance_normalization_510_56543510:7
)batch_instance_normalization_510_56543512:-
conv2d_619_56543516:
identityĒ8batch_instance_normalization_500/StatefulPartitionedCallĒ8batch_instance_normalization_501/StatefulPartitionedCallĒ8batch_instance_normalization_502/StatefulPartitionedCallĒ8batch_instance_normalization_503/StatefulPartitionedCallĒ8batch_instance_normalization_504/StatefulPartitionedCallĒ8batch_instance_normalization_505/StatefulPartitionedCallĒ8batch_instance_normalization_506/StatefulPartitionedCallĒ8batch_instance_normalization_507/StatefulPartitionedCallĒ8batch_instance_normalization_508/StatefulPartitionedCallĒ8batch_instance_normalization_509/StatefulPartitionedCallĒ8batch_instance_normalization_510/StatefulPartitionedCallĒ"conv2d_606/StatefulPartitionedCallĒ"conv2d_607/StatefulPartitionedCallĒ"conv2d_608/StatefulPartitionedCallĒ"conv2d_609/StatefulPartitionedCallĒ"conv2d_610/StatefulPartitionedCallĒ"conv2d_611/StatefulPartitionedCallĒ"conv2d_612/StatefulPartitionedCallĒ"conv2d_613/StatefulPartitionedCallĒ"conv2d_614/StatefulPartitionedCallĒ"conv2d_615/StatefulPartitionedCallĒ"conv2d_616/StatefulPartitionedCallĒ"conv2d_617/StatefulPartitionedCallĒ"conv2d_618/StatefulPartitionedCallĒ"conv2d_619/StatefulPartitionedCallĒ,conv2d_transpose_100/StatefulPartitionedCallĒ,conv2d_transpose_101/StatefulPartitionedCallY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate/concatConcatV2input_1input_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"conv2d_606/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0conv2d_606_56543376*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56541813
"conv2d_607/StatefulPartitionedCallStatefulPartitionedCall+conv2d_606/StatefulPartitionedCall:output:0conv2d_607_56543379*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56541824v
	LeakyRelu	LeakyRelu+conv2d_607/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@č
!max_pooling2d_100/PartitionedCallPartitionedCallLeakyRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56541702
"conv2d_608/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_100/PartitionedCall:output:0conv2d_608_56543384*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56541837ī
8batch_instance_normalization_500/StatefulPartitionedCallStatefulPartitionedCall+conv2d_608/StatefulPartitionedCall:output:0)batch_instance_normalization_500_56543387)batch_instance_normalization_500_56543389)batch_instance_normalization_500_56543391*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56541881
LeakyRelu_1	LeakyReluAbatch_instance_normalization_500/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_609/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_609_56543395*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56541897ī
8batch_instance_normalization_501/StatefulPartitionedCallStatefulPartitionedCall+conv2d_609/StatefulPartitionedCall:output:0)batch_instance_normalization_501_56543398)batch_instance_normalization_501_56543400)batch_instance_normalization_501_56543402*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56541941
LeakyRelu_2	LeakyReluAbatch_instance_normalization_501/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ë
!max_pooling2d_101/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56541714
"conv2d_610/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_101/PartitionedCall:output:0conv2d_610_56543407*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56541970ī
8batch_instance_normalization_502/StatefulPartitionedCallStatefulPartitionedCall+conv2d_610/StatefulPartitionedCall:output:0)batch_instance_normalization_502_56543410)batch_instance_normalization_502_56543412)batch_instance_normalization_502_56543414*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56542014
LeakyRelu_3	LeakyReluAbatch_instance_normalization_502/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_611/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_3:activations:0conv2d_611_56543418*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56542042ī
8batch_instance_normalization_503/StatefulPartitionedCallStatefulPartitionedCall+conv2d_611/StatefulPartitionedCall:output:0)batch_instance_normalization_503_56543421)batch_instance_normalization_503_56543423)batch_instance_normalization_503_56543425*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56542086
LeakyRelu_4	LeakyReluAbatch_instance_normalization_503/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_612/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_4:activations:0conv2d_612_56543429*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56542114ī
8batch_instance_normalization_504/StatefulPartitionedCallStatefulPartitionedCall+conv2d_612/StatefulPartitionedCall:output:0)batch_instance_normalization_504_56543432)batch_instance_normalization_504_56543434)batch_instance_normalization_504_56543436*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56542158
LeakyRelu_5	LeakyReluAbatch_instance_normalization_504/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_613/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_5:activations:0conv2d_613_56543440*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56542186ī
8batch_instance_normalization_505/StatefulPartitionedCallStatefulPartitionedCall+conv2d_613/StatefulPartitionedCall:output:0)batch_instance_normalization_505_56543443)batch_instance_normalization_505_56543445)batch_instance_normalization_505_56543447*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56542230
LeakyRelu_6	LeakyReluAbatch_instance_normalization_505/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ī
,conv2d_transpose_100/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_6:activations:0conv2d_transpose_100_56543451*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56541751[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
concatenate_1/concatConcatV2LeakyRelu_2:activations:05conv2d_transpose_100/StatefulPartitionedCall:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_614/StatefulPartitionedCallStatefulPartitionedCallconcatenate_1/concat:output:0conv2d_614_56543456*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56542251ī
8batch_instance_normalization_506/StatefulPartitionedCallStatefulPartitionedCall+conv2d_614/StatefulPartitionedCall:output:0)batch_instance_normalization_506_56543459)batch_instance_normalization_506_56543461)batch_instance_normalization_506_56543463*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56542295
LeakyRelu_7	LeakyReluAbatch_instance_normalization_506/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_615/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_7:activations:0conv2d_615_56543467*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56542311ī
8batch_instance_normalization_507/StatefulPartitionedCallStatefulPartitionedCall+conv2d_615/StatefulPartitionedCall:output:0)batch_instance_normalization_507_56543470)batch_instance_normalization_507_56543472)batch_instance_normalization_507_56543474*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56542355
LeakyRelu_8	LeakyReluAbatch_instance_normalization_507/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
,conv2d_transpose_101/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_8:activations:0conv2d_transpose_101_56543478*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56541790[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
concatenate_2/concatConcatV2LeakyRelu:activations:05conv2d_transpose_101/StatefulPartitionedCall:output:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
"conv2d_616/StatefulPartitionedCallStatefulPartitionedCallconcatenate_2/concat:output:0conv2d_616_56543483*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56542376ĩ
8batch_instance_normalization_508/StatefulPartitionedCallStatefulPartitionedCall+conv2d_616/StatefulPartitionedCall:output:0)batch_instance_normalization_508_56543486)batch_instance_normalization_508_56543488)batch_instance_normalization_508_56543490*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56542420
LeakyRelu_9	LeakyReluAbatch_instance_normalization_508/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
"conv2d_617/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_9:activations:0conv2d_617_56543494*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56542436ĩ
8batch_instance_normalization_509/StatefulPartitionedCallStatefulPartitionedCall+conv2d_617/StatefulPartitionedCall:output:0)batch_instance_normalization_509_56543497)batch_instance_normalization_509_56543499)batch_instance_normalization_509_56543501*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56542480
LeakyRelu_10	LeakyReluAbatch_instance_normalization_509/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
"conv2d_618/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_10:activations:0conv2d_618_56543505*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56542496ĩ
8batch_instance_normalization_510/StatefulPartitionedCallStatefulPartitionedCall+conv2d_618/StatefulPartitionedCall:output:0)batch_instance_normalization_510_56543508)batch_instance_normalization_510_56543510)batch_instance_normalization_510_56543512*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56542540
LeakyRelu_11	LeakyReluAbatch_instance_normalization_510/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"conv2d_619/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_11:activations:0conv2d_619_56543516*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56542556u
TanhTanh+conv2d_619/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸa
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸģ

NoOpNoOp9^batch_instance_normalization_500/StatefulPartitionedCall9^batch_instance_normalization_501/StatefulPartitionedCall9^batch_instance_normalization_502/StatefulPartitionedCall9^batch_instance_normalization_503/StatefulPartitionedCall9^batch_instance_normalization_504/StatefulPartitionedCall9^batch_instance_normalization_505/StatefulPartitionedCall9^batch_instance_normalization_506/StatefulPartitionedCall9^batch_instance_normalization_507/StatefulPartitionedCall9^batch_instance_normalization_508/StatefulPartitionedCall9^batch_instance_normalization_509/StatefulPartitionedCall9^batch_instance_normalization_510/StatefulPartitionedCall#^conv2d_606/StatefulPartitionedCall#^conv2d_607/StatefulPartitionedCall#^conv2d_608/StatefulPartitionedCall#^conv2d_609/StatefulPartitionedCall#^conv2d_610/StatefulPartitionedCall#^conv2d_611/StatefulPartitionedCall#^conv2d_612/StatefulPartitionedCall#^conv2d_613/StatefulPartitionedCall#^conv2d_614/StatefulPartitionedCall#^conv2d_615/StatefulPartitionedCall#^conv2d_616/StatefulPartitionedCall#^conv2d_617/StatefulPartitionedCall#^conv2d_618/StatefulPartitionedCall#^conv2d_619/StatefulPartitionedCall-^conv2d_transpose_100/StatefulPartitionedCall-^conv2d_transpose_101/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8batch_instance_normalization_500/StatefulPartitionedCall8batch_instance_normalization_500/StatefulPartitionedCall2t
8batch_instance_normalization_501/StatefulPartitionedCall8batch_instance_normalization_501/StatefulPartitionedCall2t
8batch_instance_normalization_502/StatefulPartitionedCall8batch_instance_normalization_502/StatefulPartitionedCall2t
8batch_instance_normalization_503/StatefulPartitionedCall8batch_instance_normalization_503/StatefulPartitionedCall2t
8batch_instance_normalization_504/StatefulPartitionedCall8batch_instance_normalization_504/StatefulPartitionedCall2t
8batch_instance_normalization_505/StatefulPartitionedCall8batch_instance_normalization_505/StatefulPartitionedCall2t
8batch_instance_normalization_506/StatefulPartitionedCall8batch_instance_normalization_506/StatefulPartitionedCall2t
8batch_instance_normalization_507/StatefulPartitionedCall8batch_instance_normalization_507/StatefulPartitionedCall2t
8batch_instance_normalization_508/StatefulPartitionedCall8batch_instance_normalization_508/StatefulPartitionedCall2t
8batch_instance_normalization_509/StatefulPartitionedCall8batch_instance_normalization_509/StatefulPartitionedCall2t
8batch_instance_normalization_510/StatefulPartitionedCall8batch_instance_normalization_510/StatefulPartitionedCall2H
"conv2d_606/StatefulPartitionedCall"conv2d_606/StatefulPartitionedCall2H
"conv2d_607/StatefulPartitionedCall"conv2d_607/StatefulPartitionedCall2H
"conv2d_608/StatefulPartitionedCall"conv2d_608/StatefulPartitionedCall2H
"conv2d_609/StatefulPartitionedCall"conv2d_609/StatefulPartitionedCall2H
"conv2d_610/StatefulPartitionedCall"conv2d_610/StatefulPartitionedCall2H
"conv2d_611/StatefulPartitionedCall"conv2d_611/StatefulPartitionedCall2H
"conv2d_612/StatefulPartitionedCall"conv2d_612/StatefulPartitionedCall2H
"conv2d_613/StatefulPartitionedCall"conv2d_613/StatefulPartitionedCall2H
"conv2d_614/StatefulPartitionedCall"conv2d_614/StatefulPartitionedCall2H
"conv2d_615/StatefulPartitionedCall"conv2d_615/StatefulPartitionedCall2H
"conv2d_616/StatefulPartitionedCall"conv2d_616/StatefulPartitionedCall2H
"conv2d_617/StatefulPartitionedCall"conv2d_617/StatefulPartitionedCall2H
"conv2d_618/StatefulPartitionedCall"conv2d_618/StatefulPartitionedCall2H
"conv2d_619/StatefulPartitionedCall"conv2d_619/StatefulPartitionedCall2\
,conv2d_transpose_100/StatefulPartitionedCall,conv2d_transpose_100/StatefulPartitionedCall2\
,conv2d_transpose_101/StatefulPartitionedCall,conv2d_transpose_101/StatefulPartitionedCall:Z V
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_2
Š
ŧ
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56542311

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
Ó

-__inference_conv2d_612_layer_call_fn_56545413

inputs#
unknown:
identityĒStatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56542114x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs

k
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56545252

inputs
identityĒ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ:r n
J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs

Í
,__inference_face_g_18_layer_call_fn_56543880
inputs_0
inputs_1!
unknown:@#
	unknown_0:@@$
	unknown_1:@
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	&

unknown_13:

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:&

unknown_26:

unknown_27:	

unknown_28:	

unknown_29:	&

unknown_30:

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:@%

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@$

unknown_39:@@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45:

unknown_46:$

unknown_47:
identityĒStatefulPartitionedCallĸ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543165y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"
_user_specified_name
inputs/1
·$
Î
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56542355
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ@@w
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
:ĸĸĸĸĸĸĸĸĸ@@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ@@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@e
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
:ĸĸĸĸĸĸĸĸĸ@@_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex

Ë
,__inference_face_g_18_layer_call_fn_56543370
input_1
input_2!
unknown:@#
	unknown_0:@@$
	unknown_1:@
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	&

unknown_13:

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:&

unknown_26:

unknown_27:	

unknown_28:	

unknown_29:	&

unknown_30:

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:@%

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@$

unknown_39:@@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45:

unknown_46:$

unknown_47:
identityĒStatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543165y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_2
Ø

-__inference_conv2d_616_layer_call_fn_56545771

inputs"
unknown:@
identityĒStatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56542376y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ĸĸĸĸĸĸĸĸĸ: 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
ëģ
ï4
G__inference_face_g_18_layer_call_and_return_conditional_losses_56544968
inputs_0
inputs_1C
)conv2d_606_conv2d_readvariableop_resource:@C
)conv2d_607_conv2d_readvariableop_resource:@@D
)conv2d_608_conv2d_readvariableop_resource:@G
8batch_instance_normalization_500_readvariableop_resource:	M
>batch_instance_normalization_500_mul_4_readvariableop_resource:	M
>batch_instance_normalization_500_add_3_readvariableop_resource:	E
)conv2d_609_conv2d_readvariableop_resource:G
8batch_instance_normalization_501_readvariableop_resource:	M
>batch_instance_normalization_501_mul_4_readvariableop_resource:	M
>batch_instance_normalization_501_add_3_readvariableop_resource:	E
)conv2d_610_conv2d_readvariableop_resource:G
8batch_instance_normalization_502_readvariableop_resource:	M
>batch_instance_normalization_502_mul_4_readvariableop_resource:	M
>batch_instance_normalization_502_add_3_readvariableop_resource:	E
)conv2d_611_conv2d_readvariableop_resource:G
8batch_instance_normalization_503_readvariableop_resource:	M
>batch_instance_normalization_503_mul_4_readvariableop_resource:	M
>batch_instance_normalization_503_add_3_readvariableop_resource:	E
)conv2d_612_conv2d_readvariableop_resource:G
8batch_instance_normalization_504_readvariableop_resource:	M
>batch_instance_normalization_504_mul_4_readvariableop_resource:	M
>batch_instance_normalization_504_add_3_readvariableop_resource:	E
)conv2d_613_conv2d_readvariableop_resource:G
8batch_instance_normalization_505_readvariableop_resource:	M
>batch_instance_normalization_505_mul_4_readvariableop_resource:	M
>batch_instance_normalization_505_add_3_readvariableop_resource:	Y
=conv2d_transpose_100_conv2d_transpose_readvariableop_resource:E
)conv2d_614_conv2d_readvariableop_resource:G
8batch_instance_normalization_506_readvariableop_resource:	M
>batch_instance_normalization_506_mul_4_readvariableop_resource:	M
>batch_instance_normalization_506_add_3_readvariableop_resource:	E
)conv2d_615_conv2d_readvariableop_resource:G
8batch_instance_normalization_507_readvariableop_resource:	M
>batch_instance_normalization_507_mul_4_readvariableop_resource:	M
>batch_instance_normalization_507_add_3_readvariableop_resource:	X
=conv2d_transpose_101_conv2d_transpose_readvariableop_resource:@D
)conv2d_616_conv2d_readvariableop_resource:@F
8batch_instance_normalization_508_readvariableop_resource:@L
>batch_instance_normalization_508_mul_4_readvariableop_resource:@L
>batch_instance_normalization_508_add_3_readvariableop_resource:@C
)conv2d_617_conv2d_readvariableop_resource:@@F
8batch_instance_normalization_509_readvariableop_resource:@L
>batch_instance_normalization_509_mul_4_readvariableop_resource:@L
>batch_instance_normalization_509_add_3_readvariableop_resource:@C
)conv2d_618_conv2d_readvariableop_resource:@F
8batch_instance_normalization_510_readvariableop_resource:L
>batch_instance_normalization_510_mul_4_readvariableop_resource:L
>batch_instance_normalization_510_add_3_readvariableop_resource:C
)conv2d_619_conv2d_readvariableop_resource:
identityĒ/batch_instance_normalization_500/ReadVariableOpĒ1batch_instance_normalization_500/ReadVariableOp_1Ē5batch_instance_normalization_500/add_3/ReadVariableOpĒ5batch_instance_normalization_500/mul_4/ReadVariableOpĒ/batch_instance_normalization_501/ReadVariableOpĒ1batch_instance_normalization_501/ReadVariableOp_1Ē5batch_instance_normalization_501/add_3/ReadVariableOpĒ5batch_instance_normalization_501/mul_4/ReadVariableOpĒ/batch_instance_normalization_502/ReadVariableOpĒ1batch_instance_normalization_502/ReadVariableOp_1Ē5batch_instance_normalization_502/add_3/ReadVariableOpĒ5batch_instance_normalization_502/mul_4/ReadVariableOpĒ/batch_instance_normalization_503/ReadVariableOpĒ1batch_instance_normalization_503/ReadVariableOp_1Ē5batch_instance_normalization_503/add_3/ReadVariableOpĒ5batch_instance_normalization_503/mul_4/ReadVariableOpĒ/batch_instance_normalization_504/ReadVariableOpĒ1batch_instance_normalization_504/ReadVariableOp_1Ē5batch_instance_normalization_504/add_3/ReadVariableOpĒ5batch_instance_normalization_504/mul_4/ReadVariableOpĒ/batch_instance_normalization_505/ReadVariableOpĒ1batch_instance_normalization_505/ReadVariableOp_1Ē5batch_instance_normalization_505/add_3/ReadVariableOpĒ5batch_instance_normalization_505/mul_4/ReadVariableOpĒ/batch_instance_normalization_506/ReadVariableOpĒ1batch_instance_normalization_506/ReadVariableOp_1Ē5batch_instance_normalization_506/add_3/ReadVariableOpĒ5batch_instance_normalization_506/mul_4/ReadVariableOpĒ/batch_instance_normalization_507/ReadVariableOpĒ1batch_instance_normalization_507/ReadVariableOp_1Ē5batch_instance_normalization_507/add_3/ReadVariableOpĒ5batch_instance_normalization_507/mul_4/ReadVariableOpĒ/batch_instance_normalization_508/ReadVariableOpĒ1batch_instance_normalization_508/ReadVariableOp_1Ē5batch_instance_normalization_508/add_3/ReadVariableOpĒ5batch_instance_normalization_508/mul_4/ReadVariableOpĒ/batch_instance_normalization_509/ReadVariableOpĒ1batch_instance_normalization_509/ReadVariableOp_1Ē5batch_instance_normalization_509/add_3/ReadVariableOpĒ5batch_instance_normalization_509/mul_4/ReadVariableOpĒ/batch_instance_normalization_510/ReadVariableOpĒ1batch_instance_normalization_510/ReadVariableOp_1Ē5batch_instance_normalization_510/add_3/ReadVariableOpĒ5batch_instance_normalization_510/mul_4/ReadVariableOpĒ conv2d_606/Conv2D/ReadVariableOpĒ conv2d_607/Conv2D/ReadVariableOpĒ conv2d_608/Conv2D/ReadVariableOpĒ conv2d_609/Conv2D/ReadVariableOpĒ conv2d_610/Conv2D/ReadVariableOpĒ conv2d_611/Conv2D/ReadVariableOpĒ conv2d_612/Conv2D/ReadVariableOpĒ conv2d_613/Conv2D/ReadVariableOpĒ conv2d_614/Conv2D/ReadVariableOpĒ conv2d_615/Conv2D/ReadVariableOpĒ conv2d_616/Conv2D/ReadVariableOpĒ conv2d_617/Conv2D/ReadVariableOpĒ conv2d_618/Conv2D/ReadVariableOpĒ conv2d_619/Conv2D/ReadVariableOpĒ4conv2d_transpose_100/conv2d_transpose/ReadVariableOpĒ4conv2d_transpose_101/conv2d_transpose/ReadVariableOpY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_606/Conv2D/ReadVariableOpReadVariableOp)conv2d_606_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Æ
conv2d_606/Conv2DConv2Dconcatenate/concat:output:0(conv2d_606/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

 conv2d_607/Conv2D/ReadVariableOpReadVariableOp)conv2d_607_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Å
conv2d_607/Conv2DConv2Dconv2d_606/Conv2D:output:0(conv2d_607/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
e
	LeakyRelu	LeakyReluconv2d_607/Conv2D:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Š
max_pooling2d_100/MaxPoolMaxPoolLeakyRelu:activations:0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@*
ksize
*
paddingVALID*
strides

 conv2d_608/Conv2D/ReadVariableOpReadVariableOp)conv2d_608_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ė
conv2d_608/Conv2DConv2D"max_pooling2d_100/MaxPool:output:0(conv2d_608/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

?batch_instance_normalization_500/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Þ
-batch_instance_normalization_500/moments/meanMeanconv2d_608/Conv2D:output:0Hbatch_instance_normalization_500/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_500/moments/StopGradientStopGradient6batch_instance_normalization_500/moments/mean:output:0*
T0*'
_output_shapes
:æ
:batch_instance_normalization_500/moments/SquaredDifferenceSquaredDifferenceconv2d_608/Conv2D:output:0>batch_instance_normalization_500/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Cbatch_instance_normalization_500/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_500/moments/varianceMean>batch_instance_normalization_500/moments/SquaredDifference:z:0Lbatch_instance_normalization_500/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(š
$batch_instance_normalization_500/subSubconv2d_608/Conv2D:output:06batch_instance_normalization_500/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@k
&batch_instance_normalization_500/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_500/addAddV2:batch_instance_normalization_500/moments/variance:output:0/batch_instance_normalization_500/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_500/RsqrtRsqrt(batch_instance_normalization_500/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_500/mulMul(batch_instance_normalization_500/sub:z:0*batch_instance_normalization_500/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Abatch_instance_normalization_500/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ë
/batch_instance_normalization_500/moments_1/meanMeanconv2d_608/Conv2D:output:0Jbatch_instance_normalization_500/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_500/moments_1/StopGradientStopGradient8batch_instance_normalization_500/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸę
<batch_instance_normalization_500/moments_1/SquaredDifferenceSquaredDifferenceconv2d_608/Conv2D:output:0@batch_instance_normalization_500/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Ebatch_instance_normalization_500/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_500/moments_1/varianceMean@batch_instance_normalization_500/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_500/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ū
&batch_instance_normalization_500/sub_1Subconv2d_608/Conv2D:output:08batch_instance_normalization_500/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@m
(batch_instance_normalization_500/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_500/add_1AddV2<batch_instance_normalization_500/moments_1/variance:output:01batch_instance_normalization_500/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_500/Rsqrt_1Rsqrt*batch_instance_normalization_500/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_500/mul_1Mul*batch_instance_normalization_500/sub_1:z:0,batch_instance_normalization_500/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
/batch_instance_normalization_500/ReadVariableOpReadVariableOp8batch_instance_normalization_500_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_500/mul_2Mul7batch_instance_normalization_500/ReadVariableOp:value:0(batch_instance_normalization_500/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@§
1batch_instance_normalization_500/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_500_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_500/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_500/sub_2Sub1batch_instance_normalization_500/sub_2/x:output:09batch_instance_normalization_500/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_500/mul_3Mul*batch_instance_normalization_500/sub_2:z:0*batch_instance_normalization_500/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Â
&batch_instance_normalization_500/add_2AddV2*batch_instance_normalization_500/mul_2:z:0*batch_instance_normalization_500/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_500/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_500_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_500/mul_4Mul*batch_instance_normalization_500/add_2:z:0=batch_instance_normalization_500/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_500/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_500_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_500/add_3AddV2*batch_instance_normalization_500/mul_4:z:0=batch_instance_normalization_500/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@v
LeakyRelu_1	LeakyRelu*batch_instance_normalization_500/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 conv2d_609/Conv2D/ReadVariableOpReadVariableOp)conv2d_609_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ã
conv2d_609/Conv2DConv2DLeakyRelu_1:activations:0(conv2d_609/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

?batch_instance_normalization_501/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Þ
-batch_instance_normalization_501/moments/meanMeanconv2d_609/Conv2D:output:0Hbatch_instance_normalization_501/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_501/moments/StopGradientStopGradient6batch_instance_normalization_501/moments/mean:output:0*
T0*'
_output_shapes
:æ
:batch_instance_normalization_501/moments/SquaredDifferenceSquaredDifferenceconv2d_609/Conv2D:output:0>batch_instance_normalization_501/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Cbatch_instance_normalization_501/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_501/moments/varianceMean>batch_instance_normalization_501/moments/SquaredDifference:z:0Lbatch_instance_normalization_501/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(š
$batch_instance_normalization_501/subSubconv2d_609/Conv2D:output:06batch_instance_normalization_501/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@k
&batch_instance_normalization_501/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_501/addAddV2:batch_instance_normalization_501/moments/variance:output:0/batch_instance_normalization_501/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_501/RsqrtRsqrt(batch_instance_normalization_501/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_501/mulMul(batch_instance_normalization_501/sub:z:0*batch_instance_normalization_501/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Abatch_instance_normalization_501/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ë
/batch_instance_normalization_501/moments_1/meanMeanconv2d_609/Conv2D:output:0Jbatch_instance_normalization_501/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_501/moments_1/StopGradientStopGradient8batch_instance_normalization_501/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸę
<batch_instance_normalization_501/moments_1/SquaredDifferenceSquaredDifferenceconv2d_609/Conv2D:output:0@batch_instance_normalization_501/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Ebatch_instance_normalization_501/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_501/moments_1/varianceMean@batch_instance_normalization_501/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_501/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ū
&batch_instance_normalization_501/sub_1Subconv2d_609/Conv2D:output:08batch_instance_normalization_501/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@m
(batch_instance_normalization_501/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_501/add_1AddV2<batch_instance_normalization_501/moments_1/variance:output:01batch_instance_normalization_501/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_501/Rsqrt_1Rsqrt*batch_instance_normalization_501/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_501/mul_1Mul*batch_instance_normalization_501/sub_1:z:0,batch_instance_normalization_501/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
/batch_instance_normalization_501/ReadVariableOpReadVariableOp8batch_instance_normalization_501_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_501/mul_2Mul7batch_instance_normalization_501/ReadVariableOp:value:0(batch_instance_normalization_501/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@§
1batch_instance_normalization_501/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_501_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_501/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_501/sub_2Sub1batch_instance_normalization_501/sub_2/x:output:09batch_instance_normalization_501/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_501/mul_3Mul*batch_instance_normalization_501/sub_2:z:0*batch_instance_normalization_501/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Â
&batch_instance_normalization_501/add_2AddV2*batch_instance_normalization_501/mul_2:z:0*batch_instance_normalization_501/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_501/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_501_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_501/mul_4Mul*batch_instance_normalization_501/add_2:z:0=batch_instance_normalization_501/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_501/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_501_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_501/add_3AddV2*batch_instance_normalization_501/mul_4:z:0=batch_instance_normalization_501/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@v
LeakyRelu_2	LeakyRelu*batch_instance_normalization_501/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@­
max_pooling2d_101/MaxPoolMaxPoolLeakyRelu_2:activations:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *
ksize
*
paddingVALID*
strides
p
conv2d_610/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      w
conv2d_610/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            x
conv2d_610/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
>conv2d_610/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
;conv2d_610/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
8conv2d_610/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                }
,conv2d_610/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)conv2d_610/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ü
 conv2d_610/Conv2D/SpaceToBatchNDSpaceToBatchND"max_pooling2d_101/MaxPool:output:05conv2d_610/Conv2D/SpaceToBatchND/block_shape:output:02conv2d_610/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_610/Conv2D/ReadVariableOpReadVariableOp)conv2d_610_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_610/Conv2DConv2D)conv2d_610/Conv2D/SpaceToBatchND:output:0(conv2d_610/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
}
,conv2d_610/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
&conv2d_610/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                ņ
 conv2d_610/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_610/Conv2D:output:05conv2d_610/Conv2D/BatchToSpaceND/block_shape:output:0/conv2d_610/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
?batch_instance_normalization_502/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          í
-batch_instance_normalization_502/moments/meanMean)conv2d_610/Conv2D/BatchToSpaceND:output:0Hbatch_instance_normalization_502/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_502/moments/StopGradientStopGradient6batch_instance_normalization_502/moments/mean:output:0*
T0*'
_output_shapes
:õ
:batch_instance_normalization_502/moments/SquaredDifferenceSquaredDifference)conv2d_610/Conv2D/BatchToSpaceND:output:0>batch_instance_normalization_502/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Cbatch_instance_normalization_502/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_502/moments/varianceMean>batch_instance_normalization_502/moments/SquaredDifference:z:0Lbatch_instance_normalization_502/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(É
$batch_instance_normalization_502/subSub)conv2d_610/Conv2D/BatchToSpaceND:output:06batch_instance_normalization_502/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  k
&batch_instance_normalization_502/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_502/addAddV2:batch_instance_normalization_502/moments/variance:output:0/batch_instance_normalization_502/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_502/RsqrtRsqrt(batch_instance_normalization_502/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_502/mulMul(batch_instance_normalization_502/sub:z:0*batch_instance_normalization_502/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Abatch_instance_normalization_502/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ú
/batch_instance_normalization_502/moments_1/meanMean)conv2d_610/Conv2D/BatchToSpaceND:output:0Jbatch_instance_normalization_502/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_502/moments_1/StopGradientStopGradient8batch_instance_normalization_502/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸų
<batch_instance_normalization_502/moments_1/SquaredDifferenceSquaredDifference)conv2d_610/Conv2D/BatchToSpaceND:output:0@batch_instance_normalization_502/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Ebatch_instance_normalization_502/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_502/moments_1/varianceMean@batch_instance_normalization_502/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_502/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Í
&batch_instance_normalization_502/sub_1Sub)conv2d_610/Conv2D/BatchToSpaceND:output:08batch_instance_normalization_502/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  m
(batch_instance_normalization_502/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_502/add_1AddV2<batch_instance_normalization_502/moments_1/variance:output:01batch_instance_normalization_502/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_502/Rsqrt_1Rsqrt*batch_instance_normalization_502/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_502/mul_1Mul*batch_instance_normalization_502/sub_1:z:0,batch_instance_normalization_502/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ĩ
/batch_instance_normalization_502/ReadVariableOpReadVariableOp8batch_instance_normalization_502_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_502/mul_2Mul7batch_instance_normalization_502/ReadVariableOp:value:0(batch_instance_normalization_502/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  §
1batch_instance_normalization_502/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_502_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_502/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_502/sub_2Sub1batch_instance_normalization_502/sub_2/x:output:09batch_instance_normalization_502/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_502/mul_3Mul*batch_instance_normalization_502/sub_2:z:0*batch_instance_normalization_502/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Â
&batch_instance_normalization_502/add_2AddV2*batch_instance_normalization_502/mul_2:z:0*batch_instance_normalization_502/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_502/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_502_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_502/mul_4Mul*batch_instance_normalization_502/add_2:z:0=batch_instance_normalization_502/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_502/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_502_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_502/add_3AddV2*batch_instance_normalization_502/mul_4:z:0=batch_instance_normalization_502/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
LeakyRelu_3	LeakyRelu*batch_instance_normalization_502/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  p
conv2d_611/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      w
conv2d_611/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            x
conv2d_611/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
>conv2d_611/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
;conv2d_611/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
8conv2d_611/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                }
,conv2d_611/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)conv2d_611/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ó
 conv2d_611/Conv2D/SpaceToBatchNDSpaceToBatchNDLeakyRelu_3:activations:05conv2d_611/Conv2D/SpaceToBatchND/block_shape:output:02conv2d_611/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ


 conv2d_611/Conv2D/ReadVariableOpReadVariableOp)conv2d_611_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_611/Conv2DConv2D)conv2d_611/Conv2D/SpaceToBatchND:output:0(conv2d_611/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
}
,conv2d_611/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
&conv2d_611/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                ņ
 conv2d_611/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_611/Conv2D:output:05conv2d_611/Conv2D/BatchToSpaceND/block_shape:output:0/conv2d_611/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
?batch_instance_normalization_503/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          í
-batch_instance_normalization_503/moments/meanMean)conv2d_611/Conv2D/BatchToSpaceND:output:0Hbatch_instance_normalization_503/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_503/moments/StopGradientStopGradient6batch_instance_normalization_503/moments/mean:output:0*
T0*'
_output_shapes
:õ
:batch_instance_normalization_503/moments/SquaredDifferenceSquaredDifference)conv2d_611/Conv2D/BatchToSpaceND:output:0>batch_instance_normalization_503/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Cbatch_instance_normalization_503/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_503/moments/varianceMean>batch_instance_normalization_503/moments/SquaredDifference:z:0Lbatch_instance_normalization_503/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(É
$batch_instance_normalization_503/subSub)conv2d_611/Conv2D/BatchToSpaceND:output:06batch_instance_normalization_503/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  k
&batch_instance_normalization_503/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_503/addAddV2:batch_instance_normalization_503/moments/variance:output:0/batch_instance_normalization_503/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_503/RsqrtRsqrt(batch_instance_normalization_503/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_503/mulMul(batch_instance_normalization_503/sub:z:0*batch_instance_normalization_503/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Abatch_instance_normalization_503/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ú
/batch_instance_normalization_503/moments_1/meanMean)conv2d_611/Conv2D/BatchToSpaceND:output:0Jbatch_instance_normalization_503/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_503/moments_1/StopGradientStopGradient8batch_instance_normalization_503/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸų
<batch_instance_normalization_503/moments_1/SquaredDifferenceSquaredDifference)conv2d_611/Conv2D/BatchToSpaceND:output:0@batch_instance_normalization_503/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Ebatch_instance_normalization_503/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_503/moments_1/varianceMean@batch_instance_normalization_503/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_503/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Í
&batch_instance_normalization_503/sub_1Sub)conv2d_611/Conv2D/BatchToSpaceND:output:08batch_instance_normalization_503/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  m
(batch_instance_normalization_503/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_503/add_1AddV2<batch_instance_normalization_503/moments_1/variance:output:01batch_instance_normalization_503/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_503/Rsqrt_1Rsqrt*batch_instance_normalization_503/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_503/mul_1Mul*batch_instance_normalization_503/sub_1:z:0,batch_instance_normalization_503/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ĩ
/batch_instance_normalization_503/ReadVariableOpReadVariableOp8batch_instance_normalization_503_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_503/mul_2Mul7batch_instance_normalization_503/ReadVariableOp:value:0(batch_instance_normalization_503/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  §
1batch_instance_normalization_503/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_503_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_503/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_503/sub_2Sub1batch_instance_normalization_503/sub_2/x:output:09batch_instance_normalization_503/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_503/mul_3Mul*batch_instance_normalization_503/sub_2:z:0*batch_instance_normalization_503/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Â
&batch_instance_normalization_503/add_2AddV2*batch_instance_normalization_503/mul_2:z:0*batch_instance_normalization_503/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_503/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_503_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_503/mul_4Mul*batch_instance_normalization_503/add_2:z:0=batch_instance_normalization_503/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_503/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_503_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_503/add_3AddV2*batch_instance_normalization_503/mul_4:z:0=batch_instance_normalization_503/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
LeakyRelu_4	LeakyRelu*batch_instance_normalization_503/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  p
conv2d_612/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      w
conv2d_612/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            x
conv2d_612/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
>conv2d_612/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
;conv2d_612/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
8conv2d_612/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                }
,conv2d_612/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)conv2d_612/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ó
 conv2d_612/Conv2D/SpaceToBatchNDSpaceToBatchNDLeakyRelu_4:activations:05conv2d_612/Conv2D/SpaceToBatchND/block_shape:output:02conv2d_612/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_612/Conv2D/ReadVariableOpReadVariableOp)conv2d_612_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_612/Conv2DConv2D)conv2d_612/Conv2D/SpaceToBatchND:output:0(conv2d_612/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
}
,conv2d_612/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
&conv2d_612/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                ņ
 conv2d_612/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_612/Conv2D:output:05conv2d_612/Conv2D/BatchToSpaceND/block_shape:output:0/conv2d_612/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
?batch_instance_normalization_504/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          í
-batch_instance_normalization_504/moments/meanMean)conv2d_612/Conv2D/BatchToSpaceND:output:0Hbatch_instance_normalization_504/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_504/moments/StopGradientStopGradient6batch_instance_normalization_504/moments/mean:output:0*
T0*'
_output_shapes
:õ
:batch_instance_normalization_504/moments/SquaredDifferenceSquaredDifference)conv2d_612/Conv2D/BatchToSpaceND:output:0>batch_instance_normalization_504/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Cbatch_instance_normalization_504/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_504/moments/varianceMean>batch_instance_normalization_504/moments/SquaredDifference:z:0Lbatch_instance_normalization_504/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(É
$batch_instance_normalization_504/subSub)conv2d_612/Conv2D/BatchToSpaceND:output:06batch_instance_normalization_504/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  k
&batch_instance_normalization_504/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_504/addAddV2:batch_instance_normalization_504/moments/variance:output:0/batch_instance_normalization_504/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_504/RsqrtRsqrt(batch_instance_normalization_504/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_504/mulMul(batch_instance_normalization_504/sub:z:0*batch_instance_normalization_504/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Abatch_instance_normalization_504/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ú
/batch_instance_normalization_504/moments_1/meanMean)conv2d_612/Conv2D/BatchToSpaceND:output:0Jbatch_instance_normalization_504/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_504/moments_1/StopGradientStopGradient8batch_instance_normalization_504/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸų
<batch_instance_normalization_504/moments_1/SquaredDifferenceSquaredDifference)conv2d_612/Conv2D/BatchToSpaceND:output:0@batch_instance_normalization_504/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Ebatch_instance_normalization_504/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_504/moments_1/varianceMean@batch_instance_normalization_504/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_504/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Í
&batch_instance_normalization_504/sub_1Sub)conv2d_612/Conv2D/BatchToSpaceND:output:08batch_instance_normalization_504/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  m
(batch_instance_normalization_504/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_504/add_1AddV2<batch_instance_normalization_504/moments_1/variance:output:01batch_instance_normalization_504/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_504/Rsqrt_1Rsqrt*batch_instance_normalization_504/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_504/mul_1Mul*batch_instance_normalization_504/sub_1:z:0,batch_instance_normalization_504/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ĩ
/batch_instance_normalization_504/ReadVariableOpReadVariableOp8batch_instance_normalization_504_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_504/mul_2Mul7batch_instance_normalization_504/ReadVariableOp:value:0(batch_instance_normalization_504/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  §
1batch_instance_normalization_504/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_504_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_504/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_504/sub_2Sub1batch_instance_normalization_504/sub_2/x:output:09batch_instance_normalization_504/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_504/mul_3Mul*batch_instance_normalization_504/sub_2:z:0*batch_instance_normalization_504/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Â
&batch_instance_normalization_504/add_2AddV2*batch_instance_normalization_504/mul_2:z:0*batch_instance_normalization_504/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_504/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_504_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_504/mul_4Mul*batch_instance_normalization_504/add_2:z:0=batch_instance_normalization_504/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_504/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_504_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_504/add_3AddV2*batch_instance_normalization_504/mul_4:z:0=batch_instance_normalization_504/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
LeakyRelu_5	LeakyRelu*batch_instance_normalization_504/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  p
conv2d_613/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      w
conv2d_613/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            x
conv2d_613/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
>conv2d_613/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
;conv2d_613/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
8conv2d_613/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                }
,conv2d_613/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)conv2d_613/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ó
 conv2d_613/Conv2D/SpaceToBatchNDSpaceToBatchNDLeakyRelu_5:activations:05conv2d_613/Conv2D/SpaceToBatchND/block_shape:output:02conv2d_613/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_613/Conv2D/ReadVariableOpReadVariableOp)conv2d_613_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_613/Conv2DConv2D)conv2d_613/Conv2D/SpaceToBatchND:output:0(conv2d_613/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
}
,conv2d_613/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
&conv2d_613/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                ņ
 conv2d_613/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_613/Conv2D:output:05conv2d_613/Conv2D/BatchToSpaceND/block_shape:output:0/conv2d_613/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
?batch_instance_normalization_505/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          í
-batch_instance_normalization_505/moments/meanMean)conv2d_613/Conv2D/BatchToSpaceND:output:0Hbatch_instance_normalization_505/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_505/moments/StopGradientStopGradient6batch_instance_normalization_505/moments/mean:output:0*
T0*'
_output_shapes
:õ
:batch_instance_normalization_505/moments/SquaredDifferenceSquaredDifference)conv2d_613/Conv2D/BatchToSpaceND:output:0>batch_instance_normalization_505/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Cbatch_instance_normalization_505/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_505/moments/varianceMean>batch_instance_normalization_505/moments/SquaredDifference:z:0Lbatch_instance_normalization_505/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(É
$batch_instance_normalization_505/subSub)conv2d_613/Conv2D/BatchToSpaceND:output:06batch_instance_normalization_505/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  k
&batch_instance_normalization_505/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_505/addAddV2:batch_instance_normalization_505/moments/variance:output:0/batch_instance_normalization_505/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_505/RsqrtRsqrt(batch_instance_normalization_505/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_505/mulMul(batch_instance_normalization_505/sub:z:0*batch_instance_normalization_505/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Abatch_instance_normalization_505/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ú
/batch_instance_normalization_505/moments_1/meanMean)conv2d_613/Conv2D/BatchToSpaceND:output:0Jbatch_instance_normalization_505/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_505/moments_1/StopGradientStopGradient8batch_instance_normalization_505/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸų
<batch_instance_normalization_505/moments_1/SquaredDifferenceSquaredDifference)conv2d_613/Conv2D/BatchToSpaceND:output:0@batch_instance_normalization_505/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Ebatch_instance_normalization_505/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_505/moments_1/varianceMean@batch_instance_normalization_505/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_505/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Í
&batch_instance_normalization_505/sub_1Sub)conv2d_613/Conv2D/BatchToSpaceND:output:08batch_instance_normalization_505/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  m
(batch_instance_normalization_505/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_505/add_1AddV2<batch_instance_normalization_505/moments_1/variance:output:01batch_instance_normalization_505/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_505/Rsqrt_1Rsqrt*batch_instance_normalization_505/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_505/mul_1Mul*batch_instance_normalization_505/sub_1:z:0,batch_instance_normalization_505/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ĩ
/batch_instance_normalization_505/ReadVariableOpReadVariableOp8batch_instance_normalization_505_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_505/mul_2Mul7batch_instance_normalization_505/ReadVariableOp:value:0(batch_instance_normalization_505/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  §
1batch_instance_normalization_505/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_505_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_505/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_505/sub_2Sub1batch_instance_normalization_505/sub_2/x:output:09batch_instance_normalization_505/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_505/mul_3Mul*batch_instance_normalization_505/sub_2:z:0*batch_instance_normalization_505/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Â
&batch_instance_normalization_505/add_2AddV2*batch_instance_normalization_505/mul_2:z:0*batch_instance_normalization_505/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_505/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_505_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_505/mul_4Mul*batch_instance_normalization_505/add_2:z:0=batch_instance_normalization_505/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_505/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_505_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_505/add_3AddV2*batch_instance_normalization_505/mul_4:z:0=batch_instance_normalization_505/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
LeakyRelu_6	LeakyRelu*batch_instance_normalization_505/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
conv2d_transpose_100/ShapeShapeLeakyRelu_6:activations:0*
T0*
_output_shapes
:r
(conv2d_transpose_100/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_100/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_100/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:š
"conv2d_transpose_100/strided_sliceStridedSlice#conv2d_transpose_100/Shape:output:01conv2d_transpose_100/strided_slice/stack:output:03conv2d_transpose_100/strided_slice/stack_1:output:03conv2d_transpose_100/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_100/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@^
conv2d_transpose_100/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@_
conv2d_transpose_100/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ō
conv2d_transpose_100/stackPack+conv2d_transpose_100/strided_slice:output:0%conv2d_transpose_100/stack/1:output:0%conv2d_transpose_100/stack/2:output:0%conv2d_transpose_100/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_100/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_100/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_100/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Â
$conv2d_transpose_100/strided_slice_1StridedSlice#conv2d_transpose_100/stack:output:03conv2d_transpose_100/strided_slice_1/stack:output:05conv2d_transpose_100/strided_slice_1/stack_1:output:05conv2d_transpose_100/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskž
4conv2d_transpose_100/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_100_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
%conv2d_transpose_100/conv2d_transposeConv2DBackpropInput#conv2d_transpose_100/stack:output:0<conv2d_transpose_100/conv2d_transpose/ReadVariableOp:value:0LeakyRelu_6:activations:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ó
concatenate_1/concatConcatV2LeakyRelu_2:activations:0.conv2d_transpose_100/conv2d_transpose:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 conv2d_614/Conv2D/ReadVariableOpReadVariableOp)conv2d_614_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Į
conv2d_614/Conv2DConv2Dconcatenate_1/concat:output:0(conv2d_614/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

?batch_instance_normalization_506/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Þ
-batch_instance_normalization_506/moments/meanMeanconv2d_614/Conv2D:output:0Hbatch_instance_normalization_506/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_506/moments/StopGradientStopGradient6batch_instance_normalization_506/moments/mean:output:0*
T0*'
_output_shapes
:æ
:batch_instance_normalization_506/moments/SquaredDifferenceSquaredDifferenceconv2d_614/Conv2D:output:0>batch_instance_normalization_506/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Cbatch_instance_normalization_506/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_506/moments/varianceMean>batch_instance_normalization_506/moments/SquaredDifference:z:0Lbatch_instance_normalization_506/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(š
$batch_instance_normalization_506/subSubconv2d_614/Conv2D:output:06batch_instance_normalization_506/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@k
&batch_instance_normalization_506/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_506/addAddV2:batch_instance_normalization_506/moments/variance:output:0/batch_instance_normalization_506/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_506/RsqrtRsqrt(batch_instance_normalization_506/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_506/mulMul(batch_instance_normalization_506/sub:z:0*batch_instance_normalization_506/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Abatch_instance_normalization_506/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ë
/batch_instance_normalization_506/moments_1/meanMeanconv2d_614/Conv2D:output:0Jbatch_instance_normalization_506/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_506/moments_1/StopGradientStopGradient8batch_instance_normalization_506/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸę
<batch_instance_normalization_506/moments_1/SquaredDifferenceSquaredDifferenceconv2d_614/Conv2D:output:0@batch_instance_normalization_506/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Ebatch_instance_normalization_506/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_506/moments_1/varianceMean@batch_instance_normalization_506/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_506/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ū
&batch_instance_normalization_506/sub_1Subconv2d_614/Conv2D:output:08batch_instance_normalization_506/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@m
(batch_instance_normalization_506/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_506/add_1AddV2<batch_instance_normalization_506/moments_1/variance:output:01batch_instance_normalization_506/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_506/Rsqrt_1Rsqrt*batch_instance_normalization_506/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_506/mul_1Mul*batch_instance_normalization_506/sub_1:z:0,batch_instance_normalization_506/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
/batch_instance_normalization_506/ReadVariableOpReadVariableOp8batch_instance_normalization_506_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_506/mul_2Mul7batch_instance_normalization_506/ReadVariableOp:value:0(batch_instance_normalization_506/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@§
1batch_instance_normalization_506/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_506_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_506/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_506/sub_2Sub1batch_instance_normalization_506/sub_2/x:output:09batch_instance_normalization_506/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_506/mul_3Mul*batch_instance_normalization_506/sub_2:z:0*batch_instance_normalization_506/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Â
&batch_instance_normalization_506/add_2AddV2*batch_instance_normalization_506/mul_2:z:0*batch_instance_normalization_506/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_506/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_506_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_506/mul_4Mul*batch_instance_normalization_506/add_2:z:0=batch_instance_normalization_506/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_506/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_506_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_506/add_3AddV2*batch_instance_normalization_506/mul_4:z:0=batch_instance_normalization_506/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@v
LeakyRelu_7	LeakyRelu*batch_instance_normalization_506/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 conv2d_615/Conv2D/ReadVariableOpReadVariableOp)conv2d_615_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ã
conv2d_615/Conv2DConv2DLeakyRelu_7:activations:0(conv2d_615/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

?batch_instance_normalization_507/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Þ
-batch_instance_normalization_507/moments/meanMeanconv2d_615/Conv2D:output:0Hbatch_instance_normalization_507/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_507/moments/StopGradientStopGradient6batch_instance_normalization_507/moments/mean:output:0*
T0*'
_output_shapes
:æ
:batch_instance_normalization_507/moments/SquaredDifferenceSquaredDifferenceconv2d_615/Conv2D:output:0>batch_instance_normalization_507/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Cbatch_instance_normalization_507/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_507/moments/varianceMean>batch_instance_normalization_507/moments/SquaredDifference:z:0Lbatch_instance_normalization_507/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(š
$batch_instance_normalization_507/subSubconv2d_615/Conv2D:output:06batch_instance_normalization_507/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@k
&batch_instance_normalization_507/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_507/addAddV2:batch_instance_normalization_507/moments/variance:output:0/batch_instance_normalization_507/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_507/RsqrtRsqrt(batch_instance_normalization_507/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_507/mulMul(batch_instance_normalization_507/sub:z:0*batch_instance_normalization_507/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Abatch_instance_normalization_507/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ë
/batch_instance_normalization_507/moments_1/meanMeanconv2d_615/Conv2D:output:0Jbatch_instance_normalization_507/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_507/moments_1/StopGradientStopGradient8batch_instance_normalization_507/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸę
<batch_instance_normalization_507/moments_1/SquaredDifferenceSquaredDifferenceconv2d_615/Conv2D:output:0@batch_instance_normalization_507/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Ebatch_instance_normalization_507/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_507/moments_1/varianceMean@batch_instance_normalization_507/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_507/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ū
&batch_instance_normalization_507/sub_1Subconv2d_615/Conv2D:output:08batch_instance_normalization_507/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@m
(batch_instance_normalization_507/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_507/add_1AddV2<batch_instance_normalization_507/moments_1/variance:output:01batch_instance_normalization_507/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_507/Rsqrt_1Rsqrt*batch_instance_normalization_507/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_507/mul_1Mul*batch_instance_normalization_507/sub_1:z:0,batch_instance_normalization_507/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
/batch_instance_normalization_507/ReadVariableOpReadVariableOp8batch_instance_normalization_507_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_507/mul_2Mul7batch_instance_normalization_507/ReadVariableOp:value:0(batch_instance_normalization_507/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@§
1batch_instance_normalization_507/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_507_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_507/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_507/sub_2Sub1batch_instance_normalization_507/sub_2/x:output:09batch_instance_normalization_507/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_507/mul_3Mul*batch_instance_normalization_507/sub_2:z:0*batch_instance_normalization_507/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Â
&batch_instance_normalization_507/add_2AddV2*batch_instance_normalization_507/mul_2:z:0*batch_instance_normalization_507/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_507/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_507_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_507/mul_4Mul*batch_instance_normalization_507/add_2:z:0=batch_instance_normalization_507/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_507/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_507_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_507/add_3AddV2*batch_instance_normalization_507/mul_4:z:0=batch_instance_normalization_507/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@v
LeakyRelu_8	LeakyRelu*batch_instance_normalization_507/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
conv2d_transpose_101/ShapeShapeLeakyRelu_8:activations:0*
T0*
_output_shapes
:r
(conv2d_transpose_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:š
"conv2d_transpose_101/strided_sliceStridedSlice#conv2d_transpose_101/Shape:output:01conv2d_transpose_101/strided_slice/stack:output:03conv2d_transpose_101/strided_slice/stack_1:output:03conv2d_transpose_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
conv2d_transpose_101/stack/1Const*
_output_shapes
: *
dtype0*
value
B :_
conv2d_transpose_101/stack/2Const*
_output_shapes
: *
dtype0*
value
B :^
conv2d_transpose_101/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@ō
conv2d_transpose_101/stackPack+conv2d_transpose_101/strided_slice:output:0%conv2d_transpose_101/stack/1:output:0%conv2d_transpose_101/stack/2:output:0%conv2d_transpose_101/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Â
$conv2d_transpose_101/strided_slice_1StridedSlice#conv2d_transpose_101/stack:output:03conv2d_transpose_101/strided_slice_1/stack:output:05conv2d_transpose_101/strided_slice_1/stack_1:output:05conv2d_transpose_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskŧ
4conv2d_transpose_101/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_101_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0
%conv2d_transpose_101/conv2d_transposeConv2DBackpropInput#conv2d_transpose_101/stack:output:0<conv2d_transpose_101/conv2d_transpose/ReadVariableOp:value:0LeakyRelu_8:activations:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ó
concatenate_2/concatConcatV2LeakyRelu:activations:0.conv2d_transpose_101/conv2d_transpose:output:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
 conv2d_616/Conv2D/ReadVariableOpReadVariableOp)conv2d_616_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Č
conv2d_616/Conv2DConv2Dconcatenate_2/concat:output:0(conv2d_616/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

?batch_instance_normalization_508/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ý
-batch_instance_normalization_508/moments/meanMeanconv2d_616/Conv2D:output:0Hbatch_instance_normalization_508/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(Ū
5batch_instance_normalization_508/moments/StopGradientStopGradient6batch_instance_normalization_508/moments/mean:output:0*
T0*&
_output_shapes
:@į
:batch_instance_normalization_508/moments/SquaredDifferenceSquaredDifferenceconv2d_616/Conv2D:output:0>batch_instance_normalization_508/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Cbatch_instance_normalization_508/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_508/moments/varianceMean>batch_instance_normalization_508/moments/SquaredDifference:z:0Lbatch_instance_normalization_508/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(ŧ
$batch_instance_normalization_508/subSubconv2d_616/Conv2D:output:06batch_instance_normalization_508/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@k
&batch_instance_normalization_508/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ë
$batch_instance_normalization_508/addAddV2:batch_instance_normalization_508/moments/variance:output:0/batch_instance_normalization_508/add/y:output:0*
T0*&
_output_shapes
:@
&batch_instance_normalization_508/RsqrtRsqrt(batch_instance_normalization_508/add:z:0*
T0*&
_output_shapes
:@―
$batch_instance_normalization_508/mulMul(batch_instance_normalization_508/sub:z:0*batch_instance_normalization_508/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Abatch_instance_normalization_508/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ę
/batch_instance_normalization_508/moments_1/meanMeanconv2d_616/Conv2D:output:0Jbatch_instance_normalization_508/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(ŧ
7batch_instance_normalization_508/moments_1/StopGradientStopGradient8batch_instance_normalization_508/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ë
<batch_instance_normalization_508/moments_1/SquaredDifferenceSquaredDifferenceconv2d_616/Conv2D:output:0@batch_instance_normalization_508/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Ebatch_instance_normalization_508/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_508/moments_1/varianceMean@batch_instance_normalization_508/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_508/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(ŋ
&batch_instance_normalization_508/sub_1Subconv2d_616/Conv2D:output:08batch_instance_normalization_508/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@m
(batch_instance_normalization_508/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ú
&batch_instance_normalization_508/add_1AddV2<batch_instance_normalization_508/moments_1/variance:output:01batch_instance_normalization_508/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
(batch_instance_normalization_508/Rsqrt_1Rsqrt*batch_instance_normalization_508/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ã
&batch_instance_normalization_508/mul_1Mul*batch_instance_normalization_508/sub_1:z:0,batch_instance_normalization_508/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ī
/batch_instance_normalization_508/ReadVariableOpReadVariableOp8batch_instance_normalization_508_readvariableop_resource*
_output_shapes
:@*
dtype0Ė
&batch_instance_normalization_508/mul_2Mul7batch_instance_normalization_508/ReadVariableOp:value:0(batch_instance_normalization_508/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ķ
1batch_instance_normalization_508/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_508_readvariableop_resource*
_output_shapes
:@*
dtype0m
(batch_instance_normalization_508/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ā
&batch_instance_normalization_508/sub_2Sub1batch_instance_normalization_508/sub_2/x:output:09batch_instance_normalization_508/ReadVariableOp_1:value:0*
T0*
_output_shapes
:@Á
&batch_instance_normalization_508/mul_3Mul*batch_instance_normalization_508/sub_2:z:0*batch_instance_normalization_508/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ã
&batch_instance_normalization_508/add_2AddV2*batch_instance_normalization_508/mul_2:z:0*batch_instance_normalization_508/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@°
5batch_instance_normalization_508/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_508_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0Ô
&batch_instance_normalization_508/mul_4Mul*batch_instance_normalization_508/add_2:z:0=batch_instance_normalization_508/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@°
5batch_instance_normalization_508/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_508_add_3_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
&batch_instance_normalization_508/add_3AddV2*batch_instance_normalization_508/mul_4:z:0=batch_instance_normalization_508/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@w
LeakyRelu_9	LeakyRelu*batch_instance_normalization_508/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 conv2d_617/Conv2D/ReadVariableOpReadVariableOp)conv2d_617_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ä
conv2d_617/Conv2DConv2DLeakyRelu_9:activations:0(conv2d_617/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

?batch_instance_normalization_509/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ý
-batch_instance_normalization_509/moments/meanMeanconv2d_617/Conv2D:output:0Hbatch_instance_normalization_509/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(Ū
5batch_instance_normalization_509/moments/StopGradientStopGradient6batch_instance_normalization_509/moments/mean:output:0*
T0*&
_output_shapes
:@į
:batch_instance_normalization_509/moments/SquaredDifferenceSquaredDifferenceconv2d_617/Conv2D:output:0>batch_instance_normalization_509/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Cbatch_instance_normalization_509/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_509/moments/varianceMean>batch_instance_normalization_509/moments/SquaredDifference:z:0Lbatch_instance_normalization_509/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(ŧ
$batch_instance_normalization_509/subSubconv2d_617/Conv2D:output:06batch_instance_normalization_509/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@k
&batch_instance_normalization_509/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ë
$batch_instance_normalization_509/addAddV2:batch_instance_normalization_509/moments/variance:output:0/batch_instance_normalization_509/add/y:output:0*
T0*&
_output_shapes
:@
&batch_instance_normalization_509/RsqrtRsqrt(batch_instance_normalization_509/add:z:0*
T0*&
_output_shapes
:@―
$batch_instance_normalization_509/mulMul(batch_instance_normalization_509/sub:z:0*batch_instance_normalization_509/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Abatch_instance_normalization_509/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ę
/batch_instance_normalization_509/moments_1/meanMeanconv2d_617/Conv2D:output:0Jbatch_instance_normalization_509/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(ŧ
7batch_instance_normalization_509/moments_1/StopGradientStopGradient8batch_instance_normalization_509/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ë
<batch_instance_normalization_509/moments_1/SquaredDifferenceSquaredDifferenceconv2d_617/Conv2D:output:0@batch_instance_normalization_509/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Ebatch_instance_normalization_509/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_509/moments_1/varianceMean@batch_instance_normalization_509/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_509/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(ŋ
&batch_instance_normalization_509/sub_1Subconv2d_617/Conv2D:output:08batch_instance_normalization_509/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@m
(batch_instance_normalization_509/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ú
&batch_instance_normalization_509/add_1AddV2<batch_instance_normalization_509/moments_1/variance:output:01batch_instance_normalization_509/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
(batch_instance_normalization_509/Rsqrt_1Rsqrt*batch_instance_normalization_509/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ã
&batch_instance_normalization_509/mul_1Mul*batch_instance_normalization_509/sub_1:z:0,batch_instance_normalization_509/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ī
/batch_instance_normalization_509/ReadVariableOpReadVariableOp8batch_instance_normalization_509_readvariableop_resource*
_output_shapes
:@*
dtype0Ė
&batch_instance_normalization_509/mul_2Mul7batch_instance_normalization_509/ReadVariableOp:value:0(batch_instance_normalization_509/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ķ
1batch_instance_normalization_509/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_509_readvariableop_resource*
_output_shapes
:@*
dtype0m
(batch_instance_normalization_509/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ā
&batch_instance_normalization_509/sub_2Sub1batch_instance_normalization_509/sub_2/x:output:09batch_instance_normalization_509/ReadVariableOp_1:value:0*
T0*
_output_shapes
:@Á
&batch_instance_normalization_509/mul_3Mul*batch_instance_normalization_509/sub_2:z:0*batch_instance_normalization_509/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ã
&batch_instance_normalization_509/add_2AddV2*batch_instance_normalization_509/mul_2:z:0*batch_instance_normalization_509/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@°
5batch_instance_normalization_509/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_509_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0Ô
&batch_instance_normalization_509/mul_4Mul*batch_instance_normalization_509/add_2:z:0=batch_instance_normalization_509/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@°
5batch_instance_normalization_509/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_509_add_3_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
&batch_instance_normalization_509/add_3AddV2*batch_instance_normalization_509/mul_4:z:0=batch_instance_normalization_509/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@x
LeakyRelu_10	LeakyRelu*batch_instance_normalization_509/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 conv2d_618/Conv2D/ReadVariableOpReadVariableOp)conv2d_618_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Å
conv2d_618/Conv2DConv2DLeakyRelu_10:activations:0(conv2d_618/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides

?batch_instance_normalization_510/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ý
-batch_instance_normalization_510/moments/meanMeanconv2d_618/Conv2D:output:0Hbatch_instance_normalization_510/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(Ū
5batch_instance_normalization_510/moments/StopGradientStopGradient6batch_instance_normalization_510/moments/mean:output:0*
T0*&
_output_shapes
:į
:batch_instance_normalization_510/moments/SquaredDifferenceSquaredDifferenceconv2d_618/Conv2D:output:0>batch_instance_normalization_510/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Cbatch_instance_normalization_510/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_510/moments/varianceMean>batch_instance_normalization_510/moments/SquaredDifference:z:0Lbatch_instance_normalization_510/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(ŧ
$batch_instance_normalization_510/subSubconv2d_618/Conv2D:output:06batch_instance_normalization_510/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸk
&batch_instance_normalization_510/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ë
$batch_instance_normalization_510/addAddV2:batch_instance_normalization_510/moments/variance:output:0/batch_instance_normalization_510/add/y:output:0*
T0*&
_output_shapes
:
&batch_instance_normalization_510/RsqrtRsqrt(batch_instance_normalization_510/add:z:0*
T0*&
_output_shapes
:―
$batch_instance_normalization_510/mulMul(batch_instance_normalization_510/sub:z:0*batch_instance_normalization_510/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Abatch_instance_normalization_510/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ę
/batch_instance_normalization_510/moments_1/meanMeanconv2d_618/Conv2D:output:0Jbatch_instance_normalization_510/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ŧ
7batch_instance_normalization_510/moments_1/StopGradientStopGradient8batch_instance_normalization_510/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸë
<batch_instance_normalization_510/moments_1/SquaredDifferenceSquaredDifferenceconv2d_618/Conv2D:output:0@batch_instance_normalization_510/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Ebatch_instance_normalization_510/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_510/moments_1/varianceMean@batch_instance_normalization_510/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_510/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ŋ
&batch_instance_normalization_510/sub_1Subconv2d_618/Conv2D:output:08batch_instance_normalization_510/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸm
(batch_instance_normalization_510/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ú
&batch_instance_normalization_510/add_1AddV2<batch_instance_normalization_510/moments_1/variance:output:01batch_instance_normalization_510/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_510/Rsqrt_1Rsqrt*batch_instance_normalization_510/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸÃ
&batch_instance_normalization_510/mul_1Mul*batch_instance_normalization_510/sub_1:z:0,batch_instance_normalization_510/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸĪ
/batch_instance_normalization_510/ReadVariableOpReadVariableOp8batch_instance_normalization_510_readvariableop_resource*
_output_shapes
:*
dtype0Ė
&batch_instance_normalization_510/mul_2Mul7batch_instance_normalization_510/ReadVariableOp:value:0(batch_instance_normalization_510/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸĶ
1batch_instance_normalization_510/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_510_readvariableop_resource*
_output_shapes
:*
dtype0m
(batch_instance_normalization_510/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ā
&batch_instance_normalization_510/sub_2Sub1batch_instance_normalization_510/sub_2/x:output:09batch_instance_normalization_510/ReadVariableOp_1:value:0*
T0*
_output_shapes
:Á
&batch_instance_normalization_510/mul_3Mul*batch_instance_normalization_510/sub_2:z:0*batch_instance_normalization_510/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸÃ
&batch_instance_normalization_510/add_2AddV2*batch_instance_normalization_510/mul_2:z:0*batch_instance_normalization_510/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ°
5batch_instance_normalization_510/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_510_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0Ô
&batch_instance_normalization_510/mul_4Mul*batch_instance_normalization_510/add_2:z:0=batch_instance_normalization_510/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ°
5batch_instance_normalization_510/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_510_add_3_readvariableop_resource*
_output_shapes
:*
dtype0Ö
&batch_instance_normalization_510/add_3AddV2*batch_instance_normalization_510/mul_4:z:0=batch_instance_normalization_510/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸx
LeakyRelu_11	LeakyRelu*batch_instance_normalization_510/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_619/Conv2D/ReadVariableOpReadVariableOp)conv2d_619_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Å
conv2d_619/Conv2DConv2DLeakyRelu_11:activations:0(conv2d_619/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides
d
TanhTanhconv2d_619/Conv2D:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸa
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸÐ
NoOpNoOp0^batch_instance_normalization_500/ReadVariableOp2^batch_instance_normalization_500/ReadVariableOp_16^batch_instance_normalization_500/add_3/ReadVariableOp6^batch_instance_normalization_500/mul_4/ReadVariableOp0^batch_instance_normalization_501/ReadVariableOp2^batch_instance_normalization_501/ReadVariableOp_16^batch_instance_normalization_501/add_3/ReadVariableOp6^batch_instance_normalization_501/mul_4/ReadVariableOp0^batch_instance_normalization_502/ReadVariableOp2^batch_instance_normalization_502/ReadVariableOp_16^batch_instance_normalization_502/add_3/ReadVariableOp6^batch_instance_normalization_502/mul_4/ReadVariableOp0^batch_instance_normalization_503/ReadVariableOp2^batch_instance_normalization_503/ReadVariableOp_16^batch_instance_normalization_503/add_3/ReadVariableOp6^batch_instance_normalization_503/mul_4/ReadVariableOp0^batch_instance_normalization_504/ReadVariableOp2^batch_instance_normalization_504/ReadVariableOp_16^batch_instance_normalization_504/add_3/ReadVariableOp6^batch_instance_normalization_504/mul_4/ReadVariableOp0^batch_instance_normalization_505/ReadVariableOp2^batch_instance_normalization_505/ReadVariableOp_16^batch_instance_normalization_505/add_3/ReadVariableOp6^batch_instance_normalization_505/mul_4/ReadVariableOp0^batch_instance_normalization_506/ReadVariableOp2^batch_instance_normalization_506/ReadVariableOp_16^batch_instance_normalization_506/add_3/ReadVariableOp6^batch_instance_normalization_506/mul_4/ReadVariableOp0^batch_instance_normalization_507/ReadVariableOp2^batch_instance_normalization_507/ReadVariableOp_16^batch_instance_normalization_507/add_3/ReadVariableOp6^batch_instance_normalization_507/mul_4/ReadVariableOp0^batch_instance_normalization_508/ReadVariableOp2^batch_instance_normalization_508/ReadVariableOp_16^batch_instance_normalization_508/add_3/ReadVariableOp6^batch_instance_normalization_508/mul_4/ReadVariableOp0^batch_instance_normalization_509/ReadVariableOp2^batch_instance_normalization_509/ReadVariableOp_16^batch_instance_normalization_509/add_3/ReadVariableOp6^batch_instance_normalization_509/mul_4/ReadVariableOp0^batch_instance_normalization_510/ReadVariableOp2^batch_instance_normalization_510/ReadVariableOp_16^batch_instance_normalization_510/add_3/ReadVariableOp6^batch_instance_normalization_510/mul_4/ReadVariableOp!^conv2d_606/Conv2D/ReadVariableOp!^conv2d_607/Conv2D/ReadVariableOp!^conv2d_608/Conv2D/ReadVariableOp!^conv2d_609/Conv2D/ReadVariableOp!^conv2d_610/Conv2D/ReadVariableOp!^conv2d_611/Conv2D/ReadVariableOp!^conv2d_612/Conv2D/ReadVariableOp!^conv2d_613/Conv2D/ReadVariableOp!^conv2d_614/Conv2D/ReadVariableOp!^conv2d_615/Conv2D/ReadVariableOp!^conv2d_616/Conv2D/ReadVariableOp!^conv2d_617/Conv2D/ReadVariableOp!^conv2d_618/Conv2D/ReadVariableOp!^conv2d_619/Conv2D/ReadVariableOp5^conv2d_transpose_100/conv2d_transpose/ReadVariableOp5^conv2d_transpose_101/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_instance_normalization_500/ReadVariableOp/batch_instance_normalization_500/ReadVariableOp2f
1batch_instance_normalization_500/ReadVariableOp_11batch_instance_normalization_500/ReadVariableOp_12n
5batch_instance_normalization_500/add_3/ReadVariableOp5batch_instance_normalization_500/add_3/ReadVariableOp2n
5batch_instance_normalization_500/mul_4/ReadVariableOp5batch_instance_normalization_500/mul_4/ReadVariableOp2b
/batch_instance_normalization_501/ReadVariableOp/batch_instance_normalization_501/ReadVariableOp2f
1batch_instance_normalization_501/ReadVariableOp_11batch_instance_normalization_501/ReadVariableOp_12n
5batch_instance_normalization_501/add_3/ReadVariableOp5batch_instance_normalization_501/add_3/ReadVariableOp2n
5batch_instance_normalization_501/mul_4/ReadVariableOp5batch_instance_normalization_501/mul_4/ReadVariableOp2b
/batch_instance_normalization_502/ReadVariableOp/batch_instance_normalization_502/ReadVariableOp2f
1batch_instance_normalization_502/ReadVariableOp_11batch_instance_normalization_502/ReadVariableOp_12n
5batch_instance_normalization_502/add_3/ReadVariableOp5batch_instance_normalization_502/add_3/ReadVariableOp2n
5batch_instance_normalization_502/mul_4/ReadVariableOp5batch_instance_normalization_502/mul_4/ReadVariableOp2b
/batch_instance_normalization_503/ReadVariableOp/batch_instance_normalization_503/ReadVariableOp2f
1batch_instance_normalization_503/ReadVariableOp_11batch_instance_normalization_503/ReadVariableOp_12n
5batch_instance_normalization_503/add_3/ReadVariableOp5batch_instance_normalization_503/add_3/ReadVariableOp2n
5batch_instance_normalization_503/mul_4/ReadVariableOp5batch_instance_normalization_503/mul_4/ReadVariableOp2b
/batch_instance_normalization_504/ReadVariableOp/batch_instance_normalization_504/ReadVariableOp2f
1batch_instance_normalization_504/ReadVariableOp_11batch_instance_normalization_504/ReadVariableOp_12n
5batch_instance_normalization_504/add_3/ReadVariableOp5batch_instance_normalization_504/add_3/ReadVariableOp2n
5batch_instance_normalization_504/mul_4/ReadVariableOp5batch_instance_normalization_504/mul_4/ReadVariableOp2b
/batch_instance_normalization_505/ReadVariableOp/batch_instance_normalization_505/ReadVariableOp2f
1batch_instance_normalization_505/ReadVariableOp_11batch_instance_normalization_505/ReadVariableOp_12n
5batch_instance_normalization_505/add_3/ReadVariableOp5batch_instance_normalization_505/add_3/ReadVariableOp2n
5batch_instance_normalization_505/mul_4/ReadVariableOp5batch_instance_normalization_505/mul_4/ReadVariableOp2b
/batch_instance_normalization_506/ReadVariableOp/batch_instance_normalization_506/ReadVariableOp2f
1batch_instance_normalization_506/ReadVariableOp_11batch_instance_normalization_506/ReadVariableOp_12n
5batch_instance_normalization_506/add_3/ReadVariableOp5batch_instance_normalization_506/add_3/ReadVariableOp2n
5batch_instance_normalization_506/mul_4/ReadVariableOp5batch_instance_normalization_506/mul_4/ReadVariableOp2b
/batch_instance_normalization_507/ReadVariableOp/batch_instance_normalization_507/ReadVariableOp2f
1batch_instance_normalization_507/ReadVariableOp_11batch_instance_normalization_507/ReadVariableOp_12n
5batch_instance_normalization_507/add_3/ReadVariableOp5batch_instance_normalization_507/add_3/ReadVariableOp2n
5batch_instance_normalization_507/mul_4/ReadVariableOp5batch_instance_normalization_507/mul_4/ReadVariableOp2b
/batch_instance_normalization_508/ReadVariableOp/batch_instance_normalization_508/ReadVariableOp2f
1batch_instance_normalization_508/ReadVariableOp_11batch_instance_normalization_508/ReadVariableOp_12n
5batch_instance_normalization_508/add_3/ReadVariableOp5batch_instance_normalization_508/add_3/ReadVariableOp2n
5batch_instance_normalization_508/mul_4/ReadVariableOp5batch_instance_normalization_508/mul_4/ReadVariableOp2b
/batch_instance_normalization_509/ReadVariableOp/batch_instance_normalization_509/ReadVariableOp2f
1batch_instance_normalization_509/ReadVariableOp_11batch_instance_normalization_509/ReadVariableOp_12n
5batch_instance_normalization_509/add_3/ReadVariableOp5batch_instance_normalization_509/add_3/ReadVariableOp2n
5batch_instance_normalization_509/mul_4/ReadVariableOp5batch_instance_normalization_509/mul_4/ReadVariableOp2b
/batch_instance_normalization_510/ReadVariableOp/batch_instance_normalization_510/ReadVariableOp2f
1batch_instance_normalization_510/ReadVariableOp_11batch_instance_normalization_510/ReadVariableOp_12n
5batch_instance_normalization_510/add_3/ReadVariableOp5batch_instance_normalization_510/add_3/ReadVariableOp2n
5batch_instance_normalization_510/mul_4/ReadVariableOp5batch_instance_normalization_510/mul_4/ReadVariableOp2D
 conv2d_606/Conv2D/ReadVariableOp conv2d_606/Conv2D/ReadVariableOp2D
 conv2d_607/Conv2D/ReadVariableOp conv2d_607/Conv2D/ReadVariableOp2D
 conv2d_608/Conv2D/ReadVariableOp conv2d_608/Conv2D/ReadVariableOp2D
 conv2d_609/Conv2D/ReadVariableOp conv2d_609/Conv2D/ReadVariableOp2D
 conv2d_610/Conv2D/ReadVariableOp conv2d_610/Conv2D/ReadVariableOp2D
 conv2d_611/Conv2D/ReadVariableOp conv2d_611/Conv2D/ReadVariableOp2D
 conv2d_612/Conv2D/ReadVariableOp conv2d_612/Conv2D/ReadVariableOp2D
 conv2d_613/Conv2D/ReadVariableOp conv2d_613/Conv2D/ReadVariableOp2D
 conv2d_614/Conv2D/ReadVariableOp conv2d_614/Conv2D/ReadVariableOp2D
 conv2d_615/Conv2D/ReadVariableOp conv2d_615/Conv2D/ReadVariableOp2D
 conv2d_616/Conv2D/ReadVariableOp conv2d_616/Conv2D/ReadVariableOp2D
 conv2d_617/Conv2D/ReadVariableOp conv2d_617/Conv2D/ReadVariableOp2D
 conv2d_618/Conv2D/ReadVariableOp conv2d_618/Conv2D/ReadVariableOp2D
 conv2d_619/Conv2D/ReadVariableOp conv2d_619/Conv2D/ReadVariableOp2l
4conv2d_transpose_100/conv2d_transpose/ReadVariableOp4conv2d_transpose_100/conv2d_transpose/ReadVariableOp2l
4conv2d_transpose_101/conv2d_transpose/ReadVariableOp4conv2d_transpose_101/conv2d_transpose/ReadVariableOp:[ W
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"
_user_specified_name
inputs/1
ģ
ŧ
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56541970

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOpe
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      l
Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            m
Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                r
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ŋ
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ģ
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
r
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      |
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                Å
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
IdentityIdentityConv2D/BatchToSpaceND:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
ĩ
Ã
C__inference_batch_instance_normalization_506_layer_call_fn_56545622
x
unknown:	
	unknown_0:	
	unknown_1:	
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56542295x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
Ķ
š
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56545126

inputs9
conv2d_readvariableop_resource:@
identityĒConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ĸĸĸĸĸĸĸĸĸ@@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@
 
_user_specified_nameinputs
ģ
ŧ
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56545432

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOpe
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      l
Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            m
Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                r
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ŋ
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ģ
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
r
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      |
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                Å
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
IdentityIdentityConv2D/BatchToSpaceND:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
ģ
ŧ
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56545509

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOpe
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      l
Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            m
Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                r
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ŋ
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ģ
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
r
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      |
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                Å
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
IdentityIdentityConv2D/BatchToSpaceND:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
Š
đ
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56545088

inputs8
conv2d_readvariableop_resource:@
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
­

7__inference_conv2d_transpose_101_layer_call_fn_56545734

inputs"
unknown:@
identityĒStatefulPartitionedCallũ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56541790
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ: 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
ģ
ŧ
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56545355

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOpe
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      l
Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            m
Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                r
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ŋ
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ

~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ģ
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
r
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      |
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                Å
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
IdentityIdentityConv2D/BatchToSpaceND:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
ģ$
Ë
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56542420
x%
readvariableop_resource:@+
mul_4_readvariableop_resource:@+
add_3_readvariableop_resource:@
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ķ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(`
subSubxmoments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7h
addAddV2moments/variance:output:0add/y:output:0*
T0*&
_output_shapes
:@H
RsqrtRsqrtadd:z:0*
T0*&
_output_shapes
:@Z
mulMulsub:z:0	Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(y
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ĩ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(d
sub_1Subxmoments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7w
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@U
Rsqrt_1Rsqrt	add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0i
mul_2MulReadVariableOp:value:0mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes
:@^
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:@*
dtype0q
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:@*
dtype0s
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@b
IdentityIdentity	add_3:z:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@

_user_specified_namex
ģ
ŧ
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56542186

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOpe
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      l
Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            m
Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                r
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ŋ
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ģ
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
r
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      |
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                Å
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
IdentityIdentityConv2D/BatchToSpaceND:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
Ó

-__inference_conv2d_615_layer_call_fn_56545669

inputs#
unknown:
identityĒStatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56542311x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
Š
đ
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56541813

inputs8
conv2d_readvariableop_resource:@
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Š
ŧ
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56542251

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
ģ$
Ë
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56542540
x%
readvariableop_resource:+
mul_4_readvariableop_resource:+
add_3_readvariableop_resource:
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ķ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(`
subSubxmoments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸJ
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7h
addAddV2moments/variance:output:0add/y:output:0*
T0*&
_output_shapes
:H
RsqrtRsqrtadd:z:0*
T0*&
_output_shapes
:Z
mulMulsub:z:0	Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸq
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(y
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸu
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ĩ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(d
sub_1Subxmoments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7w
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸU
Rsqrt_1Rsqrt	add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0i
mul_2MulReadVariableOp:value:0mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸd
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes
:^
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸn
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:*
dtype0q
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0s
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸb
IdentityIdentity	add_3:z:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ

_user_specified_namex
ëŋ
é
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543165

inputs
inputs_1-
conv2d_606_56543020:@-
conv2d_607_56543023:@@.
conv2d_608_56543028:@8
)batch_instance_normalization_500_56543031:	8
)batch_instance_normalization_500_56543033:	8
)batch_instance_normalization_500_56543035:	/
conv2d_609_56543039:8
)batch_instance_normalization_501_56543042:	8
)batch_instance_normalization_501_56543044:	8
)batch_instance_normalization_501_56543046:	/
conv2d_610_56543051:8
)batch_instance_normalization_502_56543054:	8
)batch_instance_normalization_502_56543056:	8
)batch_instance_normalization_502_56543058:	/
conv2d_611_56543062:8
)batch_instance_normalization_503_56543065:	8
)batch_instance_normalization_503_56543067:	8
)batch_instance_normalization_503_56543069:	/
conv2d_612_56543073:8
)batch_instance_normalization_504_56543076:	8
)batch_instance_normalization_504_56543078:	8
)batch_instance_normalization_504_56543080:	/
conv2d_613_56543084:8
)batch_instance_normalization_505_56543087:	8
)batch_instance_normalization_505_56543089:	8
)batch_instance_normalization_505_56543091:	9
conv2d_transpose_100_56543095:/
conv2d_614_56543100:8
)batch_instance_normalization_506_56543103:	8
)batch_instance_normalization_506_56543105:	8
)batch_instance_normalization_506_56543107:	/
conv2d_615_56543111:8
)batch_instance_normalization_507_56543114:	8
)batch_instance_normalization_507_56543116:	8
)batch_instance_normalization_507_56543118:	8
conv2d_transpose_101_56543122:@.
conv2d_616_56543127:@7
)batch_instance_normalization_508_56543130:@7
)batch_instance_normalization_508_56543132:@7
)batch_instance_normalization_508_56543134:@-
conv2d_617_56543138:@@7
)batch_instance_normalization_509_56543141:@7
)batch_instance_normalization_509_56543143:@7
)batch_instance_normalization_509_56543145:@-
conv2d_618_56543149:@7
)batch_instance_normalization_510_56543152:7
)batch_instance_normalization_510_56543154:7
)batch_instance_normalization_510_56543156:-
conv2d_619_56543160:
identityĒ8batch_instance_normalization_500/StatefulPartitionedCallĒ8batch_instance_normalization_501/StatefulPartitionedCallĒ8batch_instance_normalization_502/StatefulPartitionedCallĒ8batch_instance_normalization_503/StatefulPartitionedCallĒ8batch_instance_normalization_504/StatefulPartitionedCallĒ8batch_instance_normalization_505/StatefulPartitionedCallĒ8batch_instance_normalization_506/StatefulPartitionedCallĒ8batch_instance_normalization_507/StatefulPartitionedCallĒ8batch_instance_normalization_508/StatefulPartitionedCallĒ8batch_instance_normalization_509/StatefulPartitionedCallĒ8batch_instance_normalization_510/StatefulPartitionedCallĒ"conv2d_606/StatefulPartitionedCallĒ"conv2d_607/StatefulPartitionedCallĒ"conv2d_608/StatefulPartitionedCallĒ"conv2d_609/StatefulPartitionedCallĒ"conv2d_610/StatefulPartitionedCallĒ"conv2d_611/StatefulPartitionedCallĒ"conv2d_612/StatefulPartitionedCallĒ"conv2d_613/StatefulPartitionedCallĒ"conv2d_614/StatefulPartitionedCallĒ"conv2d_615/StatefulPartitionedCallĒ"conv2d_616/StatefulPartitionedCallĒ"conv2d_617/StatefulPartitionedCallĒ"conv2d_618/StatefulPartitionedCallĒ"conv2d_619/StatefulPartitionedCallĒ,conv2d_transpose_100/StatefulPartitionedCallĒ,conv2d_transpose_101/StatefulPartitionedCallY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate/concatConcatV2inputsinputs_1 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"conv2d_606/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0conv2d_606_56543020*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56541813
"conv2d_607/StatefulPartitionedCallStatefulPartitionedCall+conv2d_606/StatefulPartitionedCall:output:0conv2d_607_56543023*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56541824v
	LeakyRelu	LeakyRelu+conv2d_607/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@č
!max_pooling2d_100/PartitionedCallPartitionedCallLeakyRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56541702
"conv2d_608/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_100/PartitionedCall:output:0conv2d_608_56543028*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56541837ī
8batch_instance_normalization_500/StatefulPartitionedCallStatefulPartitionedCall+conv2d_608/StatefulPartitionedCall:output:0)batch_instance_normalization_500_56543031)batch_instance_normalization_500_56543033)batch_instance_normalization_500_56543035*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56541881
LeakyRelu_1	LeakyReluAbatch_instance_normalization_500/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_609/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_609_56543039*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56541897ī
8batch_instance_normalization_501/StatefulPartitionedCallStatefulPartitionedCall+conv2d_609/StatefulPartitionedCall:output:0)batch_instance_normalization_501_56543042)batch_instance_normalization_501_56543044)batch_instance_normalization_501_56543046*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56541941
LeakyRelu_2	LeakyReluAbatch_instance_normalization_501/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ë
!max_pooling2d_101/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56541714
"conv2d_610/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_101/PartitionedCall:output:0conv2d_610_56543051*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56541970ī
8batch_instance_normalization_502/StatefulPartitionedCallStatefulPartitionedCall+conv2d_610/StatefulPartitionedCall:output:0)batch_instance_normalization_502_56543054)batch_instance_normalization_502_56543056)batch_instance_normalization_502_56543058*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56542014
LeakyRelu_3	LeakyReluAbatch_instance_normalization_502/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_611/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_3:activations:0conv2d_611_56543062*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56542042ī
8batch_instance_normalization_503/StatefulPartitionedCallStatefulPartitionedCall+conv2d_611/StatefulPartitionedCall:output:0)batch_instance_normalization_503_56543065)batch_instance_normalization_503_56543067)batch_instance_normalization_503_56543069*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56542086
LeakyRelu_4	LeakyReluAbatch_instance_normalization_503/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_612/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_4:activations:0conv2d_612_56543073*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56542114ī
8batch_instance_normalization_504/StatefulPartitionedCallStatefulPartitionedCall+conv2d_612/StatefulPartitionedCall:output:0)batch_instance_normalization_504_56543076)batch_instance_normalization_504_56543078)batch_instance_normalization_504_56543080*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56542158
LeakyRelu_5	LeakyReluAbatch_instance_normalization_504/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_613/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_5:activations:0conv2d_613_56543084*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56542186ī
8batch_instance_normalization_505/StatefulPartitionedCallStatefulPartitionedCall+conv2d_613/StatefulPartitionedCall:output:0)batch_instance_normalization_505_56543087)batch_instance_normalization_505_56543089)batch_instance_normalization_505_56543091*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56542230
LeakyRelu_6	LeakyReluAbatch_instance_normalization_505/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ī
,conv2d_transpose_100/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_6:activations:0conv2d_transpose_100_56543095*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56541751[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
concatenate_1/concatConcatV2LeakyRelu_2:activations:05conv2d_transpose_100/StatefulPartitionedCall:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_614/StatefulPartitionedCallStatefulPartitionedCallconcatenate_1/concat:output:0conv2d_614_56543100*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56542251ī
8batch_instance_normalization_506/StatefulPartitionedCallStatefulPartitionedCall+conv2d_614/StatefulPartitionedCall:output:0)batch_instance_normalization_506_56543103)batch_instance_normalization_506_56543105)batch_instance_normalization_506_56543107*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56542295
LeakyRelu_7	LeakyReluAbatch_instance_normalization_506/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_615/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_7:activations:0conv2d_615_56543111*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56542311ī
8batch_instance_normalization_507/StatefulPartitionedCallStatefulPartitionedCall+conv2d_615/StatefulPartitionedCall:output:0)batch_instance_normalization_507_56543114)batch_instance_normalization_507_56543116)batch_instance_normalization_507_56543118*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56542355
LeakyRelu_8	LeakyReluAbatch_instance_normalization_507/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
,conv2d_transpose_101/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_8:activations:0conv2d_transpose_101_56543122*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56541790[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
concatenate_2/concatConcatV2LeakyRelu:activations:05conv2d_transpose_101/StatefulPartitionedCall:output:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
"conv2d_616/StatefulPartitionedCallStatefulPartitionedCallconcatenate_2/concat:output:0conv2d_616_56543127*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56542376ĩ
8batch_instance_normalization_508/StatefulPartitionedCallStatefulPartitionedCall+conv2d_616/StatefulPartitionedCall:output:0)batch_instance_normalization_508_56543130)batch_instance_normalization_508_56543132)batch_instance_normalization_508_56543134*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56542420
LeakyRelu_9	LeakyReluAbatch_instance_normalization_508/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
"conv2d_617/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_9:activations:0conv2d_617_56543138*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56542436ĩ
8batch_instance_normalization_509/StatefulPartitionedCallStatefulPartitionedCall+conv2d_617/StatefulPartitionedCall:output:0)batch_instance_normalization_509_56543141)batch_instance_normalization_509_56543143)batch_instance_normalization_509_56543145*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56542480
LeakyRelu_10	LeakyReluAbatch_instance_normalization_509/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
"conv2d_618/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_10:activations:0conv2d_618_56543149*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56542496ĩ
8batch_instance_normalization_510/StatefulPartitionedCallStatefulPartitionedCall+conv2d_618/StatefulPartitionedCall:output:0)batch_instance_normalization_510_56543152)batch_instance_normalization_510_56543154)batch_instance_normalization_510_56543156*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56542540
LeakyRelu_11	LeakyReluAbatch_instance_normalization_510/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"conv2d_619/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_11:activations:0conv2d_619_56543160*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56542556u
TanhTanh+conv2d_619/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸa
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸģ

NoOpNoOp9^batch_instance_normalization_500/StatefulPartitionedCall9^batch_instance_normalization_501/StatefulPartitionedCall9^batch_instance_normalization_502/StatefulPartitionedCall9^batch_instance_normalization_503/StatefulPartitionedCall9^batch_instance_normalization_504/StatefulPartitionedCall9^batch_instance_normalization_505/StatefulPartitionedCall9^batch_instance_normalization_506/StatefulPartitionedCall9^batch_instance_normalization_507/StatefulPartitionedCall9^batch_instance_normalization_508/StatefulPartitionedCall9^batch_instance_normalization_509/StatefulPartitionedCall9^batch_instance_normalization_510/StatefulPartitionedCall#^conv2d_606/StatefulPartitionedCall#^conv2d_607/StatefulPartitionedCall#^conv2d_608/StatefulPartitionedCall#^conv2d_609/StatefulPartitionedCall#^conv2d_610/StatefulPartitionedCall#^conv2d_611/StatefulPartitionedCall#^conv2d_612/StatefulPartitionedCall#^conv2d_613/StatefulPartitionedCall#^conv2d_614/StatefulPartitionedCall#^conv2d_615/StatefulPartitionedCall#^conv2d_616/StatefulPartitionedCall#^conv2d_617/StatefulPartitionedCall#^conv2d_618/StatefulPartitionedCall#^conv2d_619/StatefulPartitionedCall-^conv2d_transpose_100/StatefulPartitionedCall-^conv2d_transpose_101/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8batch_instance_normalization_500/StatefulPartitionedCall8batch_instance_normalization_500/StatefulPartitionedCall2t
8batch_instance_normalization_501/StatefulPartitionedCall8batch_instance_normalization_501/StatefulPartitionedCall2t
8batch_instance_normalization_502/StatefulPartitionedCall8batch_instance_normalization_502/StatefulPartitionedCall2t
8batch_instance_normalization_503/StatefulPartitionedCall8batch_instance_normalization_503/StatefulPartitionedCall2t
8batch_instance_normalization_504/StatefulPartitionedCall8batch_instance_normalization_504/StatefulPartitionedCall2t
8batch_instance_normalization_505/StatefulPartitionedCall8batch_instance_normalization_505/StatefulPartitionedCall2t
8batch_instance_normalization_506/StatefulPartitionedCall8batch_instance_normalization_506/StatefulPartitionedCall2t
8batch_instance_normalization_507/StatefulPartitionedCall8batch_instance_normalization_507/StatefulPartitionedCall2t
8batch_instance_normalization_508/StatefulPartitionedCall8batch_instance_normalization_508/StatefulPartitionedCall2t
8batch_instance_normalization_509/StatefulPartitionedCall8batch_instance_normalization_509/StatefulPartitionedCall2t
8batch_instance_normalization_510/StatefulPartitionedCall8batch_instance_normalization_510/StatefulPartitionedCall2H
"conv2d_606/StatefulPartitionedCall"conv2d_606/StatefulPartitionedCall2H
"conv2d_607/StatefulPartitionedCall"conv2d_607/StatefulPartitionedCall2H
"conv2d_608/StatefulPartitionedCall"conv2d_608/StatefulPartitionedCall2H
"conv2d_609/StatefulPartitionedCall"conv2d_609/StatefulPartitionedCall2H
"conv2d_610/StatefulPartitionedCall"conv2d_610/StatefulPartitionedCall2H
"conv2d_611/StatefulPartitionedCall"conv2d_611/StatefulPartitionedCall2H
"conv2d_612/StatefulPartitionedCall"conv2d_612/StatefulPartitionedCall2H
"conv2d_613/StatefulPartitionedCall"conv2d_613/StatefulPartitionedCall2H
"conv2d_614/StatefulPartitionedCall"conv2d_614/StatefulPartitionedCall2H
"conv2d_615/StatefulPartitionedCall"conv2d_615/StatefulPartitionedCall2H
"conv2d_616/StatefulPartitionedCall"conv2d_616/StatefulPartitionedCall2H
"conv2d_617/StatefulPartitionedCall"conv2d_617/StatefulPartitionedCall2H
"conv2d_618/StatefulPartitionedCall"conv2d_618/StatefulPartitionedCall2H
"conv2d_619/StatefulPartitionedCall"conv2d_619/StatefulPartitionedCall2\
,conv2d_transpose_100/StatefulPartitionedCall,conv2d_transpose_100/StatefulPartitionedCall2\
,conv2d_transpose_101/StatefulPartitionedCall,conv2d_transpose_101/StatefulPartitionedCall:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Ó

-__inference_conv2d_609_layer_call_fn_56545184

inputs#
unknown:
identityĒStatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56541897x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
íŋ
é
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543672
input_1
input_2-
conv2d_606_56543527:@-
conv2d_607_56543530:@@.
conv2d_608_56543535:@8
)batch_instance_normalization_500_56543538:	8
)batch_instance_normalization_500_56543540:	8
)batch_instance_normalization_500_56543542:	/
conv2d_609_56543546:8
)batch_instance_normalization_501_56543549:	8
)batch_instance_normalization_501_56543551:	8
)batch_instance_normalization_501_56543553:	/
conv2d_610_56543558:8
)batch_instance_normalization_502_56543561:	8
)batch_instance_normalization_502_56543563:	8
)batch_instance_normalization_502_56543565:	/
conv2d_611_56543569:8
)batch_instance_normalization_503_56543572:	8
)batch_instance_normalization_503_56543574:	8
)batch_instance_normalization_503_56543576:	/
conv2d_612_56543580:8
)batch_instance_normalization_504_56543583:	8
)batch_instance_normalization_504_56543585:	8
)batch_instance_normalization_504_56543587:	/
conv2d_613_56543591:8
)batch_instance_normalization_505_56543594:	8
)batch_instance_normalization_505_56543596:	8
)batch_instance_normalization_505_56543598:	9
conv2d_transpose_100_56543602:/
conv2d_614_56543607:8
)batch_instance_normalization_506_56543610:	8
)batch_instance_normalization_506_56543612:	8
)batch_instance_normalization_506_56543614:	/
conv2d_615_56543618:8
)batch_instance_normalization_507_56543621:	8
)batch_instance_normalization_507_56543623:	8
)batch_instance_normalization_507_56543625:	8
conv2d_transpose_101_56543629:@.
conv2d_616_56543634:@7
)batch_instance_normalization_508_56543637:@7
)batch_instance_normalization_508_56543639:@7
)batch_instance_normalization_508_56543641:@-
conv2d_617_56543645:@@7
)batch_instance_normalization_509_56543648:@7
)batch_instance_normalization_509_56543650:@7
)batch_instance_normalization_509_56543652:@-
conv2d_618_56543656:@7
)batch_instance_normalization_510_56543659:7
)batch_instance_normalization_510_56543661:7
)batch_instance_normalization_510_56543663:-
conv2d_619_56543667:
identityĒ8batch_instance_normalization_500/StatefulPartitionedCallĒ8batch_instance_normalization_501/StatefulPartitionedCallĒ8batch_instance_normalization_502/StatefulPartitionedCallĒ8batch_instance_normalization_503/StatefulPartitionedCallĒ8batch_instance_normalization_504/StatefulPartitionedCallĒ8batch_instance_normalization_505/StatefulPartitionedCallĒ8batch_instance_normalization_506/StatefulPartitionedCallĒ8batch_instance_normalization_507/StatefulPartitionedCallĒ8batch_instance_normalization_508/StatefulPartitionedCallĒ8batch_instance_normalization_509/StatefulPartitionedCallĒ8batch_instance_normalization_510/StatefulPartitionedCallĒ"conv2d_606/StatefulPartitionedCallĒ"conv2d_607/StatefulPartitionedCallĒ"conv2d_608/StatefulPartitionedCallĒ"conv2d_609/StatefulPartitionedCallĒ"conv2d_610/StatefulPartitionedCallĒ"conv2d_611/StatefulPartitionedCallĒ"conv2d_612/StatefulPartitionedCallĒ"conv2d_613/StatefulPartitionedCallĒ"conv2d_614/StatefulPartitionedCallĒ"conv2d_615/StatefulPartitionedCallĒ"conv2d_616/StatefulPartitionedCallĒ"conv2d_617/StatefulPartitionedCallĒ"conv2d_618/StatefulPartitionedCallĒ"conv2d_619/StatefulPartitionedCallĒ,conv2d_transpose_100/StatefulPartitionedCallĒ,conv2d_transpose_101/StatefulPartitionedCallY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate/concatConcatV2input_1input_2 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"conv2d_606/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0conv2d_606_56543527*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56541813
"conv2d_607/StatefulPartitionedCallStatefulPartitionedCall+conv2d_606/StatefulPartitionedCall:output:0conv2d_607_56543530*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56541824v
	LeakyRelu	LeakyRelu+conv2d_607/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@č
!max_pooling2d_100/PartitionedCallPartitionedCallLeakyRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56541702
"conv2d_608/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_100/PartitionedCall:output:0conv2d_608_56543535*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56541837ī
8batch_instance_normalization_500/StatefulPartitionedCallStatefulPartitionedCall+conv2d_608/StatefulPartitionedCall:output:0)batch_instance_normalization_500_56543538)batch_instance_normalization_500_56543540)batch_instance_normalization_500_56543542*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56541881
LeakyRelu_1	LeakyReluAbatch_instance_normalization_500/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_609/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_609_56543546*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56541897ī
8batch_instance_normalization_501/StatefulPartitionedCallStatefulPartitionedCall+conv2d_609/StatefulPartitionedCall:output:0)batch_instance_normalization_501_56543549)batch_instance_normalization_501_56543551)batch_instance_normalization_501_56543553*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56541941
LeakyRelu_2	LeakyReluAbatch_instance_normalization_501/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ë
!max_pooling2d_101/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56541714
"conv2d_610/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_101/PartitionedCall:output:0conv2d_610_56543558*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56541970ī
8batch_instance_normalization_502/StatefulPartitionedCallStatefulPartitionedCall+conv2d_610/StatefulPartitionedCall:output:0)batch_instance_normalization_502_56543561)batch_instance_normalization_502_56543563)batch_instance_normalization_502_56543565*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56542014
LeakyRelu_3	LeakyReluAbatch_instance_normalization_502/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_611/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_3:activations:0conv2d_611_56543569*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56542042ī
8batch_instance_normalization_503/StatefulPartitionedCallStatefulPartitionedCall+conv2d_611/StatefulPartitionedCall:output:0)batch_instance_normalization_503_56543572)batch_instance_normalization_503_56543574)batch_instance_normalization_503_56543576*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56542086
LeakyRelu_4	LeakyReluAbatch_instance_normalization_503/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_612/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_4:activations:0conv2d_612_56543580*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56542114ī
8batch_instance_normalization_504/StatefulPartitionedCallStatefulPartitionedCall+conv2d_612/StatefulPartitionedCall:output:0)batch_instance_normalization_504_56543583)batch_instance_normalization_504_56543585)batch_instance_normalization_504_56543587*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56542158
LeakyRelu_5	LeakyReluAbatch_instance_normalization_504/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_613/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_5:activations:0conv2d_613_56543591*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56542186ī
8batch_instance_normalization_505/StatefulPartitionedCallStatefulPartitionedCall+conv2d_613/StatefulPartitionedCall:output:0)batch_instance_normalization_505_56543594)batch_instance_normalization_505_56543596)batch_instance_normalization_505_56543598*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56542230
LeakyRelu_6	LeakyReluAbatch_instance_normalization_505/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ī
,conv2d_transpose_100/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_6:activations:0conv2d_transpose_100_56543602*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56541751[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
concatenate_1/concatConcatV2LeakyRelu_2:activations:05conv2d_transpose_100/StatefulPartitionedCall:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_614/StatefulPartitionedCallStatefulPartitionedCallconcatenate_1/concat:output:0conv2d_614_56543607*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56542251ī
8batch_instance_normalization_506/StatefulPartitionedCallStatefulPartitionedCall+conv2d_614/StatefulPartitionedCall:output:0)batch_instance_normalization_506_56543610)batch_instance_normalization_506_56543612)batch_instance_normalization_506_56543614*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56542295
LeakyRelu_7	LeakyReluAbatch_instance_normalization_506/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_615/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_7:activations:0conv2d_615_56543618*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56542311ī
8batch_instance_normalization_507/StatefulPartitionedCallStatefulPartitionedCall+conv2d_615/StatefulPartitionedCall:output:0)batch_instance_normalization_507_56543621)batch_instance_normalization_507_56543623)batch_instance_normalization_507_56543625*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56542355
LeakyRelu_8	LeakyReluAbatch_instance_normalization_507/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
,conv2d_transpose_101/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_8:activations:0conv2d_transpose_101_56543629*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56541790[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
concatenate_2/concatConcatV2LeakyRelu:activations:05conv2d_transpose_101/StatefulPartitionedCall:output:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
"conv2d_616/StatefulPartitionedCallStatefulPartitionedCallconcatenate_2/concat:output:0conv2d_616_56543634*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56542376ĩ
8batch_instance_normalization_508/StatefulPartitionedCallStatefulPartitionedCall+conv2d_616/StatefulPartitionedCall:output:0)batch_instance_normalization_508_56543637)batch_instance_normalization_508_56543639)batch_instance_normalization_508_56543641*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56542420
LeakyRelu_9	LeakyReluAbatch_instance_normalization_508/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
"conv2d_617/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_9:activations:0conv2d_617_56543645*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56542436ĩ
8batch_instance_normalization_509/StatefulPartitionedCallStatefulPartitionedCall+conv2d_617/StatefulPartitionedCall:output:0)batch_instance_normalization_509_56543648)batch_instance_normalization_509_56543650)batch_instance_normalization_509_56543652*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56542480
LeakyRelu_10	LeakyReluAbatch_instance_normalization_509/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
"conv2d_618/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_10:activations:0conv2d_618_56543656*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56542496ĩ
8batch_instance_normalization_510/StatefulPartitionedCallStatefulPartitionedCall+conv2d_618/StatefulPartitionedCall:output:0)batch_instance_normalization_510_56543659)batch_instance_normalization_510_56543661)batch_instance_normalization_510_56543663*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56542540
LeakyRelu_11	LeakyReluAbatch_instance_normalization_510/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"conv2d_619/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_11:activations:0conv2d_619_56543667*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56542556u
TanhTanh+conv2d_619/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸa
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸģ

NoOpNoOp9^batch_instance_normalization_500/StatefulPartitionedCall9^batch_instance_normalization_501/StatefulPartitionedCall9^batch_instance_normalization_502/StatefulPartitionedCall9^batch_instance_normalization_503/StatefulPartitionedCall9^batch_instance_normalization_504/StatefulPartitionedCall9^batch_instance_normalization_505/StatefulPartitionedCall9^batch_instance_normalization_506/StatefulPartitionedCall9^batch_instance_normalization_507/StatefulPartitionedCall9^batch_instance_normalization_508/StatefulPartitionedCall9^batch_instance_normalization_509/StatefulPartitionedCall9^batch_instance_normalization_510/StatefulPartitionedCall#^conv2d_606/StatefulPartitionedCall#^conv2d_607/StatefulPartitionedCall#^conv2d_608/StatefulPartitionedCall#^conv2d_609/StatefulPartitionedCall#^conv2d_610/StatefulPartitionedCall#^conv2d_611/StatefulPartitionedCall#^conv2d_612/StatefulPartitionedCall#^conv2d_613/StatefulPartitionedCall#^conv2d_614/StatefulPartitionedCall#^conv2d_615/StatefulPartitionedCall#^conv2d_616/StatefulPartitionedCall#^conv2d_617/StatefulPartitionedCall#^conv2d_618/StatefulPartitionedCall#^conv2d_619/StatefulPartitionedCall-^conv2d_transpose_100/StatefulPartitionedCall-^conv2d_transpose_101/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8batch_instance_normalization_500/StatefulPartitionedCall8batch_instance_normalization_500/StatefulPartitionedCall2t
8batch_instance_normalization_501/StatefulPartitionedCall8batch_instance_normalization_501/StatefulPartitionedCall2t
8batch_instance_normalization_502/StatefulPartitionedCall8batch_instance_normalization_502/StatefulPartitionedCall2t
8batch_instance_normalization_503/StatefulPartitionedCall8batch_instance_normalization_503/StatefulPartitionedCall2t
8batch_instance_normalization_504/StatefulPartitionedCall8batch_instance_normalization_504/StatefulPartitionedCall2t
8batch_instance_normalization_505/StatefulPartitionedCall8batch_instance_normalization_505/StatefulPartitionedCall2t
8batch_instance_normalization_506/StatefulPartitionedCall8batch_instance_normalization_506/StatefulPartitionedCall2t
8batch_instance_normalization_507/StatefulPartitionedCall8batch_instance_normalization_507/StatefulPartitionedCall2t
8batch_instance_normalization_508/StatefulPartitionedCall8batch_instance_normalization_508/StatefulPartitionedCall2t
8batch_instance_normalization_509/StatefulPartitionedCall8batch_instance_normalization_509/StatefulPartitionedCall2t
8batch_instance_normalization_510/StatefulPartitionedCall8batch_instance_normalization_510/StatefulPartitionedCall2H
"conv2d_606/StatefulPartitionedCall"conv2d_606/StatefulPartitionedCall2H
"conv2d_607/StatefulPartitionedCall"conv2d_607/StatefulPartitionedCall2H
"conv2d_608/StatefulPartitionedCall"conv2d_608/StatefulPartitionedCall2H
"conv2d_609/StatefulPartitionedCall"conv2d_609/StatefulPartitionedCall2H
"conv2d_610/StatefulPartitionedCall"conv2d_610/StatefulPartitionedCall2H
"conv2d_611/StatefulPartitionedCall"conv2d_611/StatefulPartitionedCall2H
"conv2d_612/StatefulPartitionedCall"conv2d_612/StatefulPartitionedCall2H
"conv2d_613/StatefulPartitionedCall"conv2d_613/StatefulPartitionedCall2H
"conv2d_614/StatefulPartitionedCall"conv2d_614/StatefulPartitionedCall2H
"conv2d_615/StatefulPartitionedCall"conv2d_615/StatefulPartitionedCall2H
"conv2d_616/StatefulPartitionedCall"conv2d_616/StatefulPartitionedCall2H
"conv2d_617/StatefulPartitionedCall"conv2d_617/StatefulPartitionedCall2H
"conv2d_618/StatefulPartitionedCall"conv2d_618/StatefulPartitionedCall2H
"conv2d_619/StatefulPartitionedCall"conv2d_619/StatefulPartitionedCall2\
,conv2d_transpose_100/StatefulPartitionedCall,conv2d_transpose_100/StatefulPartitionedCall2\
,conv2d_transpose_101/StatefulPartitionedCall,conv2d_transpose_101/StatefulPartitionedCall:Z V
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_2
·$
Î
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56545177
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ@@w
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
:ĸĸĸĸĸĸĸĸĸ@@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ@@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@e
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
:ĸĸĸĸĸĸĸĸĸ@@_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
Š
ŧ
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56545191

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
Š
đ
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56542496

inputs8
conv2d_readvariableop_resource:@
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
Š
đ
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56542556

inputs8
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Õ

-__inference_conv2d_607_layer_call_fn_56545095

inputs!
unknown:@@
identityĒStatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56541824y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
Â
Ø
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56545764

inputsC
(conv2d_transpose_readvariableop_resource:@
identityĒconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ņ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*A
_output_shapes/
-:+ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@h
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Õ

-__inference_conv2d_606_layer_call_fn_56545081

inputs!
unknown:@
identityĒStatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56541813y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Š
đ
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56545973

inputs8
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs

Ë
,__inference_face_g_18_layer_call_fn_56542663
input_1
input_2!
unknown:@#
	unknown_0:@@$
	unknown_1:@
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	&

unknown_13:

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:&

unknown_26:

unknown_27:	

unknown_28:	

unknown_29:	&

unknown_30:

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:@%

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@$

unknown_39:@@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45:

unknown_46:$

unknown_47:
identityĒStatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_face_g_18_layer_call_and_return_conditional_losses_56542562y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_2
Ó

-__inference_conv2d_613_layer_call_fn_56545490

inputs#
unknown:
identityĒStatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56542186x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
ģ
ŧ
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56545278

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOpe
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      l
Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            m
Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                r
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ŋ
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ģ
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
r
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      |
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                Å
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
IdentityIdentityConv2D/BatchToSpaceND:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
d
Ļ
!__inference__traced_save_56546144
file_prefix0
,savev2_conv2d_606_kernel_read_readvariableop0
,savev2_conv2d_607_kernel_read_readvariableop0
,savev2_conv2d_608_kernel_read_readvariableopC
?savev2_batch_instance_normalization_500_rho_read_readvariableopE
Asavev2_batch_instance_normalization_500_gamma_read_readvariableopD
@savev2_batch_instance_normalization_500_beta_read_readvariableop0
,savev2_conv2d_609_kernel_read_readvariableopC
?savev2_batch_instance_normalization_501_rho_read_readvariableopE
Asavev2_batch_instance_normalization_501_gamma_read_readvariableopD
@savev2_batch_instance_normalization_501_beta_read_readvariableop0
,savev2_conv2d_610_kernel_read_readvariableopC
?savev2_batch_instance_normalization_502_rho_read_readvariableopE
Asavev2_batch_instance_normalization_502_gamma_read_readvariableopD
@savev2_batch_instance_normalization_502_beta_read_readvariableop0
,savev2_conv2d_611_kernel_read_readvariableopC
?savev2_batch_instance_normalization_503_rho_read_readvariableopE
Asavev2_batch_instance_normalization_503_gamma_read_readvariableopD
@savev2_batch_instance_normalization_503_beta_read_readvariableop0
,savev2_conv2d_612_kernel_read_readvariableopC
?savev2_batch_instance_normalization_504_rho_read_readvariableopE
Asavev2_batch_instance_normalization_504_gamma_read_readvariableopD
@savev2_batch_instance_normalization_504_beta_read_readvariableop0
,savev2_conv2d_613_kernel_read_readvariableopC
?savev2_batch_instance_normalization_505_rho_read_readvariableopE
Asavev2_batch_instance_normalization_505_gamma_read_readvariableopD
@savev2_batch_instance_normalization_505_beta_read_readvariableop:
6savev2_conv2d_transpose_100_kernel_read_readvariableop0
,savev2_conv2d_614_kernel_read_readvariableopC
?savev2_batch_instance_normalization_506_rho_read_readvariableopE
Asavev2_batch_instance_normalization_506_gamma_read_readvariableopD
@savev2_batch_instance_normalization_506_beta_read_readvariableop0
,savev2_conv2d_615_kernel_read_readvariableopC
?savev2_batch_instance_normalization_507_rho_read_readvariableopE
Asavev2_batch_instance_normalization_507_gamma_read_readvariableopD
@savev2_batch_instance_normalization_507_beta_read_readvariableop:
6savev2_conv2d_transpose_101_kernel_read_readvariableop0
,savev2_conv2d_616_kernel_read_readvariableopC
?savev2_batch_instance_normalization_508_rho_read_readvariableopE
Asavev2_batch_instance_normalization_508_gamma_read_readvariableopD
@savev2_batch_instance_normalization_508_beta_read_readvariableop0
,savev2_conv2d_617_kernel_read_readvariableopC
?savev2_batch_instance_normalization_509_rho_read_readvariableopE
Asavev2_batch_instance_normalization_509_gamma_read_readvariableopD
@savev2_batch_instance_normalization_509_beta_read_readvariableop0
,savev2_conv2d_618_kernel_read_readvariableopC
?savev2_batch_instance_normalization_510_rho_read_readvariableopE
Asavev2_batch_instance_normalization_510_gamma_read_readvariableopD
@savev2_batch_instance_normalization_510_beta_read_readvariableop0
,savev2_conv2d_619_kernel_read_readvariableop
savev2_const

identity_1ĒMergeV2Checkpointsw
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
: ŧ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*ä
valueÚBŨ2B)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv1_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv2_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn2_2/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn2_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn2_2/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_2/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_2/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_3/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_3/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_3/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_3/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_4/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_4/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_4/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_4/beta/.ATTRIBUTES/VARIABLE_VALUEB(convt1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv6_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn6_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn6_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn6_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv6_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn6_2/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn6_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn6_2/beta/.ATTRIBUTES/VARIABLE_VALUEB(convt2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv7_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn7_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn7_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn7_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv7_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn7_2/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn7_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn7_2/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv7_3/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn7_3/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn7_3/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn7_3/beta/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHŅ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ũ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_606_kernel_read_readvariableop,savev2_conv2d_607_kernel_read_readvariableop,savev2_conv2d_608_kernel_read_readvariableop?savev2_batch_instance_normalization_500_rho_read_readvariableopAsavev2_batch_instance_normalization_500_gamma_read_readvariableop@savev2_batch_instance_normalization_500_beta_read_readvariableop,savev2_conv2d_609_kernel_read_readvariableop?savev2_batch_instance_normalization_501_rho_read_readvariableopAsavev2_batch_instance_normalization_501_gamma_read_readvariableop@savev2_batch_instance_normalization_501_beta_read_readvariableop,savev2_conv2d_610_kernel_read_readvariableop?savev2_batch_instance_normalization_502_rho_read_readvariableopAsavev2_batch_instance_normalization_502_gamma_read_readvariableop@savev2_batch_instance_normalization_502_beta_read_readvariableop,savev2_conv2d_611_kernel_read_readvariableop?savev2_batch_instance_normalization_503_rho_read_readvariableopAsavev2_batch_instance_normalization_503_gamma_read_readvariableop@savev2_batch_instance_normalization_503_beta_read_readvariableop,savev2_conv2d_612_kernel_read_readvariableop?savev2_batch_instance_normalization_504_rho_read_readvariableopAsavev2_batch_instance_normalization_504_gamma_read_readvariableop@savev2_batch_instance_normalization_504_beta_read_readvariableop,savev2_conv2d_613_kernel_read_readvariableop?savev2_batch_instance_normalization_505_rho_read_readvariableopAsavev2_batch_instance_normalization_505_gamma_read_readvariableop@savev2_batch_instance_normalization_505_beta_read_readvariableop6savev2_conv2d_transpose_100_kernel_read_readvariableop,savev2_conv2d_614_kernel_read_readvariableop?savev2_batch_instance_normalization_506_rho_read_readvariableopAsavev2_batch_instance_normalization_506_gamma_read_readvariableop@savev2_batch_instance_normalization_506_beta_read_readvariableop,savev2_conv2d_615_kernel_read_readvariableop?savev2_batch_instance_normalization_507_rho_read_readvariableopAsavev2_batch_instance_normalization_507_gamma_read_readvariableop@savev2_batch_instance_normalization_507_beta_read_readvariableop6savev2_conv2d_transpose_101_kernel_read_readvariableop,savev2_conv2d_616_kernel_read_readvariableop?savev2_batch_instance_normalization_508_rho_read_readvariableopAsavev2_batch_instance_normalization_508_gamma_read_readvariableop@savev2_batch_instance_normalization_508_beta_read_readvariableop,savev2_conv2d_617_kernel_read_readvariableop?savev2_batch_instance_normalization_509_rho_read_readvariableopAsavev2_batch_instance_normalization_509_gamma_read_readvariableop@savev2_batch_instance_normalization_509_beta_read_readvariableop,savev2_conv2d_618_kernel_read_readvariableop?savev2_batch_instance_normalization_510_rho_read_readvariableopAsavev2_batch_instance_normalization_510_gamma_read_readvariableop@savev2_batch_instance_normalization_510_beta_read_readvariableop,savev2_conv2d_619_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *@
dtypes6
422
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

identity_1Identity_1:output:0*Š
_input_shapes
: :@:@@:@:::::::::::::::::::::::::::::::::@:@:@:@:@:@@:@:@:@:@::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@:,(
&
_output_shapes
:@@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::. *
(
_output_shapes
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::-$)
'
_output_shapes
:@:-%)
'
_output_shapes
:@: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
:@:,)(
&
_output_shapes
:@@: *

_output_shapes
:@: +

_output_shapes
:@: ,

_output_shapes
:@:,-(
&
_output_shapes
:@: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
::,1(
&
_output_shapes
::2

_output_shapes
: 
Ū
š
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56542376

inputs9
conv2d_readvariableop_resource:@
identityĒConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ĸĸĸĸĸĸĸĸĸ: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
ĩ
Ã
C__inference_batch_instance_normalization_504_layer_call_fn_56545443
x
unknown:	
	unknown_0:	
	unknown_1:	
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56542158x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
ģ$
Ë
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56542480
x%
readvariableop_resource:@+
mul_4_readvariableop_resource:@+
add_3_readvariableop_resource:@
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ķ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(`
subSubxmoments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7h
addAddV2moments/variance:output:0add/y:output:0*
T0*&
_output_shapes
:@H
RsqrtRsqrtadd:z:0*
T0*&
_output_shapes
:@Z
mulMulsub:z:0	Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(y
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ĩ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(d
sub_1Subxmoments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7w
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@U
Rsqrt_1Rsqrt	add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0i
mul_2MulReadVariableOp:value:0mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes
:@^
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:@*
dtype0q
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:@*
dtype0s
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@b
IdentityIdentity	add_3:z:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@

_user_specified_namex

k
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56541702

inputs
identityĒ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ:r n
J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
ģ$
Ë
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56545829
x%
readvariableop_resource:@+
mul_4_readvariableop_resource:@+
add_3_readvariableop_resource:@
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ķ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(`
subSubxmoments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7h
addAddV2moments/variance:output:0add/y:output:0*
T0*&
_output_shapes
:@H
RsqrtRsqrtadd:z:0*
T0*&
_output_shapes
:@Z
mulMulsub:z:0	Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(y
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ĩ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(d
sub_1Subxmoments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7w
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@U
Rsqrt_1Rsqrt	add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0i
mul_2MulReadVariableOp:value:0mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes
:@^
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:@*
dtype0q
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:@*
dtype0s
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@b
IdentityIdentity	add_3:z:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@

_user_specified_namex
·$
Î
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56545662
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ@@w
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
:ĸĸĸĸĸĸĸĸĸ@@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ@@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@e
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
:ĸĸĸĸĸĸĸĸĸ@@_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
ĩ
Ã
C__inference_batch_instance_normalization_500_layer_call_fn_56545137
x
unknown:	
	unknown_0:	
	unknown_1:	
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56541881x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
Š
đ
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56545102

inputs8
conv2d_readvariableop_resource:@@
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
ĩ
Ã
C__inference_batch_instance_normalization_503_layer_call_fn_56545366
x
unknown:	
	unknown_0:	
	unknown_1:	
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56542086x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
âÃ
Ą$
$__inference__traced_restore_56546301
file_prefix<
"assignvariableop_conv2d_606_kernel:@>
$assignvariableop_1_conv2d_607_kernel:@@?
$assignvariableop_2_conv2d_608_kernel:@F
7assignvariableop_3_batch_instance_normalization_500_rho:	H
9assignvariableop_4_batch_instance_normalization_500_gamma:	G
8assignvariableop_5_batch_instance_normalization_500_beta:	@
$assignvariableop_6_conv2d_609_kernel:F
7assignvariableop_7_batch_instance_normalization_501_rho:	H
9assignvariableop_8_batch_instance_normalization_501_gamma:	G
8assignvariableop_9_batch_instance_normalization_501_beta:	A
%assignvariableop_10_conv2d_610_kernel:G
8assignvariableop_11_batch_instance_normalization_502_rho:	I
:assignvariableop_12_batch_instance_normalization_502_gamma:	H
9assignvariableop_13_batch_instance_normalization_502_beta:	A
%assignvariableop_14_conv2d_611_kernel:G
8assignvariableop_15_batch_instance_normalization_503_rho:	I
:assignvariableop_16_batch_instance_normalization_503_gamma:	H
9assignvariableop_17_batch_instance_normalization_503_beta:	A
%assignvariableop_18_conv2d_612_kernel:G
8assignvariableop_19_batch_instance_normalization_504_rho:	I
:assignvariableop_20_batch_instance_normalization_504_gamma:	H
9assignvariableop_21_batch_instance_normalization_504_beta:	A
%assignvariableop_22_conv2d_613_kernel:G
8assignvariableop_23_batch_instance_normalization_505_rho:	I
:assignvariableop_24_batch_instance_normalization_505_gamma:	H
9assignvariableop_25_batch_instance_normalization_505_beta:	K
/assignvariableop_26_conv2d_transpose_100_kernel:A
%assignvariableop_27_conv2d_614_kernel:G
8assignvariableop_28_batch_instance_normalization_506_rho:	I
:assignvariableop_29_batch_instance_normalization_506_gamma:	H
9assignvariableop_30_batch_instance_normalization_506_beta:	A
%assignvariableop_31_conv2d_615_kernel:G
8assignvariableop_32_batch_instance_normalization_507_rho:	I
:assignvariableop_33_batch_instance_normalization_507_gamma:	H
9assignvariableop_34_batch_instance_normalization_507_beta:	J
/assignvariableop_35_conv2d_transpose_101_kernel:@@
%assignvariableop_36_conv2d_616_kernel:@F
8assignvariableop_37_batch_instance_normalization_508_rho:@H
:assignvariableop_38_batch_instance_normalization_508_gamma:@G
9assignvariableop_39_batch_instance_normalization_508_beta:@?
%assignvariableop_40_conv2d_617_kernel:@@F
8assignvariableop_41_batch_instance_normalization_509_rho:@H
:assignvariableop_42_batch_instance_normalization_509_gamma:@G
9assignvariableop_43_batch_instance_normalization_509_beta:@?
%assignvariableop_44_conv2d_618_kernel:@F
8assignvariableop_45_batch_instance_normalization_510_rho:H
:assignvariableop_46_batch_instance_normalization_510_gamma:G
9assignvariableop_47_batch_instance_normalization_510_beta:?
%assignvariableop_48_conv2d_619_kernel:
identity_50ĒAssignVariableOpĒAssignVariableOp_1ĒAssignVariableOp_10ĒAssignVariableOp_11ĒAssignVariableOp_12ĒAssignVariableOp_13ĒAssignVariableOp_14ĒAssignVariableOp_15ĒAssignVariableOp_16ĒAssignVariableOp_17ĒAssignVariableOp_18ĒAssignVariableOp_19ĒAssignVariableOp_2ĒAssignVariableOp_20ĒAssignVariableOp_21ĒAssignVariableOp_22ĒAssignVariableOp_23ĒAssignVariableOp_24ĒAssignVariableOp_25ĒAssignVariableOp_26ĒAssignVariableOp_27ĒAssignVariableOp_28ĒAssignVariableOp_29ĒAssignVariableOp_3ĒAssignVariableOp_30ĒAssignVariableOp_31ĒAssignVariableOp_32ĒAssignVariableOp_33ĒAssignVariableOp_34ĒAssignVariableOp_35ĒAssignVariableOp_36ĒAssignVariableOp_37ĒAssignVariableOp_38ĒAssignVariableOp_39ĒAssignVariableOp_4ĒAssignVariableOp_40ĒAssignVariableOp_41ĒAssignVariableOp_42ĒAssignVariableOp_43ĒAssignVariableOp_44ĒAssignVariableOp_45ĒAssignVariableOp_46ĒAssignVariableOp_47ĒAssignVariableOp_48ĒAssignVariableOp_5ĒAssignVariableOp_6ĒAssignVariableOp_7ĒAssignVariableOp_8ĒAssignVariableOp_9ū
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*ä
valueÚBŨ2B)conv1_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv1_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv2_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn2_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn2_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn2_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv2_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn2_2/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn2_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn2_2/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_2/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_2/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_3/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_3/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_3/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_3/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv3_4/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn3_4/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn3_4/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn3_4/beta/.ATTRIBUTES/VARIABLE_VALUEB(convt1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv6_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn6_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn6_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn6_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv6_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn6_2/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn6_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn6_2/beta/.ATTRIBUTES/VARIABLE_VALUEB(convt2/kernel/.ATTRIBUTES/VARIABLE_VALUEB)conv7_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn7_1/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn7_1/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn7_1/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv7_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn7_2/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn7_2/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn7_2/beta/.ATTRIBUTES/VARIABLE_VALUEB)conv7_3/kernel/.ATTRIBUTES/VARIABLE_VALUEB$bn7_3/rho/.ATTRIBUTES/VARIABLE_VALUEB&bn7_3/gamma/.ATTRIBUTES/VARIABLE_VALUEB%bn7_3/beta/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHÔ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:2*
dtype0*w
valuenBl2B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Þ
_output_shapesË
Č::::::::::::::::::::::::::::::::::::::::::::::::::*@
dtypes6
422[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_606_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp$assignvariableop_1_conv2d_607_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_608_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ķ
AssignVariableOp_3AssignVariableOp7assignvariableop_3_batch_instance_normalization_500_rhoIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ļ
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_instance_normalization_500_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_5AssignVariableOp8assignvariableop_5_batch_instance_normalization_500_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp$assignvariableop_6_conv2d_609_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ķ
AssignVariableOp_7AssignVariableOp7assignvariableop_7_batch_instance_normalization_501_rhoIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ļ
AssignVariableOp_8AssignVariableOp9assignvariableop_8_batch_instance_normalization_501_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_instance_normalization_501_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp%assignvariableop_10_conv2d_610_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_11AssignVariableOp8assignvariableop_11_batch_instance_normalization_502_rhoIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_12AssignVariableOp:assignvariableop_12_batch_instance_normalization_502_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_instance_normalization_502_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp%assignvariableop_14_conv2d_611_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_15AssignVariableOp8assignvariableop_15_batch_instance_normalization_503_rhoIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_16AssignVariableOp:assignvariableop_16_batch_instance_normalization_503_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_instance_normalization_503_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp%assignvariableop_18_conv2d_612_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_19AssignVariableOp8assignvariableop_19_batch_instance_normalization_504_rhoIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_20AssignVariableOp:assignvariableop_20_batch_instance_normalization_504_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_instance_normalization_504_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp%assignvariableop_22_conv2d_613_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_23AssignVariableOp8assignvariableop_23_batch_instance_normalization_505_rhoIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_24AssignVariableOp:assignvariableop_24_batch_instance_normalization_505_gammaIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_25AssignVariableOp9assignvariableop_25_batch_instance_normalization_505_betaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_26AssignVariableOp/assignvariableop_26_conv2d_transpose_100_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp%assignvariableop_27_conv2d_614_kernelIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_28AssignVariableOp8assignvariableop_28_batch_instance_normalization_506_rhoIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_instance_normalization_506_gammaIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_30AssignVariableOp9assignvariableop_30_batch_instance_normalization_506_betaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp%assignvariableop_31_conv2d_615_kernelIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_32AssignVariableOp8assignvariableop_32_batch_instance_normalization_507_rhoIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_33AssignVariableOp:assignvariableop_33_batch_instance_normalization_507_gammaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_34AssignVariableOp9assignvariableop_34_batch_instance_normalization_507_betaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_35AssignVariableOp/assignvariableop_35_conv2d_transpose_101_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp%assignvariableop_36_conv2d_616_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_37AssignVariableOp8assignvariableop_37_batch_instance_normalization_508_rhoIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_38AssignVariableOp:assignvariableop_38_batch_instance_normalization_508_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_instance_normalization_508_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp%assignvariableop_40_conv2d_617_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_41AssignVariableOp8assignvariableop_41_batch_instance_normalization_509_rhoIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_42AssignVariableOp:assignvariableop_42_batch_instance_normalization_509_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_43AssignVariableOp9assignvariableop_43_batch_instance_normalization_509_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp%assignvariableop_44_conv2d_618_kernelIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Đ
AssignVariableOp_45AssignVariableOp8assignvariableop_45_batch_instance_normalization_510_rhoIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ŧ
AssignVariableOp_46AssignVariableOp:assignvariableop_46_batch_instance_normalization_510_gammaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Š
AssignVariableOp_47AssignVariableOp9assignvariableop_47_batch_instance_normalization_510_betaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp%assignvariableop_48_conv2d_619_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 	
Identity_49Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_50IdentityIdentity_49:output:0^NoOp_1*
T0*
_output_shapes
: ō
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_50Identity_50:output:0*w
_input_shapesf
d: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482(
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
ę
Å
&__inference_signature_wrapper_56545074
input_1
input_2!
unknown:@#
	unknown_0:@@$
	unknown_1:@
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	&

unknown_13:

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:&

unknown_26:

unknown_27:	

unknown_28:	

unknown_29:	&

unknown_30:

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:@%

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@$

unknown_39:@@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45:

unknown_46:$

unknown_47:
identityĒStatefulPartitionedCallŲ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_56541693y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_2
Ķ
š
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56541837

inputs9
conv2d_readvariableop_resource:@
identityĒConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ĸĸĸĸĸĸĸĸĸ@@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@
 
_user_specified_nameinputs
Ū
š
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56545778

inputs9
conv2d_readvariableop_resource:@
identityĒConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ĸĸĸĸĸĸĸĸĸ: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Z V
2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56545242
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ@@w
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
:ĸĸĸĸĸĸĸĸĸ@@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ@@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@e
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
:ĸĸĸĸĸĸĸĸĸ@@_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
Ó

-__inference_conv2d_611_layer_call_fn_56545336

inputs#
unknown:
identityĒStatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56542042x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs
Į
Ų
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56541751

inputsD
(conv2d_transpose_readvariableop_resource:
identityĒconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ņ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ý
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides

IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸh
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
ĩ
Ã
C__inference_batch_instance_normalization_501_layer_call_fn_56545202
x
unknown:	
	unknown_0:	
	unknown_1:	
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56541941x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
ģ$
Ë
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56545959
x%
readvariableop_resource:+
mul_4_readvariableop_resource:+
add_3_readvariableop_resource:
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸw
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ķ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(`
subSubxmoments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸJ
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7h
addAddV2moments/variance:output:0add/y:output:0*
T0*&
_output_shapes
:H
RsqrtRsqrtadd:z:0*
T0*&
_output_shapes
:Z
mulMulsub:z:0	Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸq
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(y
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸu
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ĩ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(d
sub_1Subxmoments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸL
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7w
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸU
Rsqrt_1Rsqrt	add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸb
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0i
mul_2MulReadVariableOp:value:0mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸd
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes
:^
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸn
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:*
dtype0q
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸn
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:*
dtype0s
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸb
IdentityIdentity	add_3:z:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ

_user_specified_namex
·$
Î
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56545329
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ  w
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
:ĸĸĸĸĸĸĸĸĸ  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  e
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
:ĸĸĸĸĸĸĸĸĸ  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
Š
đ
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56545908

inputs8
conv2d_readvariableop_resource:@
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
ķ
Ā
C__inference_batch_instance_normalization_508_layer_call_fn_56545789
x
unknown:@
	unknown_0:@
	unknown_1:@
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56542420y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@

_user_specified_namex
ëŋ
é
G__inference_face_g_18_layer_call_and_return_conditional_losses_56542562

inputs
inputs_1-
conv2d_606_56541814:@-
conv2d_607_56541825:@@.
conv2d_608_56541838:@8
)batch_instance_normalization_500_56541882:	8
)batch_instance_normalization_500_56541884:	8
)batch_instance_normalization_500_56541886:	/
conv2d_609_56541898:8
)batch_instance_normalization_501_56541942:	8
)batch_instance_normalization_501_56541944:	8
)batch_instance_normalization_501_56541946:	/
conv2d_610_56541971:8
)batch_instance_normalization_502_56542015:	8
)batch_instance_normalization_502_56542017:	8
)batch_instance_normalization_502_56542019:	/
conv2d_611_56542043:8
)batch_instance_normalization_503_56542087:	8
)batch_instance_normalization_503_56542089:	8
)batch_instance_normalization_503_56542091:	/
conv2d_612_56542115:8
)batch_instance_normalization_504_56542159:	8
)batch_instance_normalization_504_56542161:	8
)batch_instance_normalization_504_56542163:	/
conv2d_613_56542187:8
)batch_instance_normalization_505_56542231:	8
)batch_instance_normalization_505_56542233:	8
)batch_instance_normalization_505_56542235:	9
conv2d_transpose_100_56542239:/
conv2d_614_56542252:8
)batch_instance_normalization_506_56542296:	8
)batch_instance_normalization_506_56542298:	8
)batch_instance_normalization_506_56542300:	/
conv2d_615_56542312:8
)batch_instance_normalization_507_56542356:	8
)batch_instance_normalization_507_56542358:	8
)batch_instance_normalization_507_56542360:	8
conv2d_transpose_101_56542364:@.
conv2d_616_56542377:@7
)batch_instance_normalization_508_56542421:@7
)batch_instance_normalization_508_56542423:@7
)batch_instance_normalization_508_56542425:@-
conv2d_617_56542437:@@7
)batch_instance_normalization_509_56542481:@7
)batch_instance_normalization_509_56542483:@7
)batch_instance_normalization_509_56542485:@-
conv2d_618_56542497:@7
)batch_instance_normalization_510_56542541:7
)batch_instance_normalization_510_56542543:7
)batch_instance_normalization_510_56542545:-
conv2d_619_56542557:
identityĒ8batch_instance_normalization_500/StatefulPartitionedCallĒ8batch_instance_normalization_501/StatefulPartitionedCallĒ8batch_instance_normalization_502/StatefulPartitionedCallĒ8batch_instance_normalization_503/StatefulPartitionedCallĒ8batch_instance_normalization_504/StatefulPartitionedCallĒ8batch_instance_normalization_505/StatefulPartitionedCallĒ8batch_instance_normalization_506/StatefulPartitionedCallĒ8batch_instance_normalization_507/StatefulPartitionedCallĒ8batch_instance_normalization_508/StatefulPartitionedCallĒ8batch_instance_normalization_509/StatefulPartitionedCallĒ8batch_instance_normalization_510/StatefulPartitionedCallĒ"conv2d_606/StatefulPartitionedCallĒ"conv2d_607/StatefulPartitionedCallĒ"conv2d_608/StatefulPartitionedCallĒ"conv2d_609/StatefulPartitionedCallĒ"conv2d_610/StatefulPartitionedCallĒ"conv2d_611/StatefulPartitionedCallĒ"conv2d_612/StatefulPartitionedCallĒ"conv2d_613/StatefulPartitionedCallĒ"conv2d_614/StatefulPartitionedCallĒ"conv2d_615/StatefulPartitionedCallĒ"conv2d_616/StatefulPartitionedCallĒ"conv2d_617/StatefulPartitionedCallĒ"conv2d_618/StatefulPartitionedCallĒ"conv2d_619/StatefulPartitionedCallĒ,conv2d_transpose_100/StatefulPartitionedCallĒ,conv2d_transpose_101/StatefulPartitionedCallY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate/concatConcatV2inputsinputs_1 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"conv2d_606/StatefulPartitionedCallStatefulPartitionedCallconcatenate/concat:output:0conv2d_606_56541814*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56541813
"conv2d_607/StatefulPartitionedCallStatefulPartitionedCall+conv2d_606/StatefulPartitionedCall:output:0conv2d_607_56541825*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56541824v
	LeakyRelu	LeakyRelu+conv2d_607/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@č
!max_pooling2d_100/PartitionedCallPartitionedCallLeakyRelu:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56541702
"conv2d_608/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_100/PartitionedCall:output:0conv2d_608_56541838*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56541837ī
8batch_instance_normalization_500/StatefulPartitionedCallStatefulPartitionedCall+conv2d_608/StatefulPartitionedCall:output:0)batch_instance_normalization_500_56541882)batch_instance_normalization_500_56541884)batch_instance_normalization_500_56541886*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56541881
LeakyRelu_1	LeakyReluAbatch_instance_normalization_500/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_609/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_1:activations:0conv2d_609_56541898*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56541897ī
8batch_instance_normalization_501/StatefulPartitionedCallStatefulPartitionedCall+conv2d_609/StatefulPartitionedCall:output:0)batch_instance_normalization_501_56541942)batch_instance_normalization_501_56541944)batch_instance_normalization_501_56541946*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56541941
LeakyRelu_2	LeakyReluAbatch_instance_normalization_501/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ë
!max_pooling2d_101/PartitionedCallPartitionedCallLeakyRelu_2:activations:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56541714
"conv2d_610/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_101/PartitionedCall:output:0conv2d_610_56541971*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56541970ī
8batch_instance_normalization_502/StatefulPartitionedCallStatefulPartitionedCall+conv2d_610/StatefulPartitionedCall:output:0)batch_instance_normalization_502_56542015)batch_instance_normalization_502_56542017)batch_instance_normalization_502_56542019*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56542014
LeakyRelu_3	LeakyReluAbatch_instance_normalization_502/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_611/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_3:activations:0conv2d_611_56542043*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56542042ī
8batch_instance_normalization_503/StatefulPartitionedCallStatefulPartitionedCall+conv2d_611/StatefulPartitionedCall:output:0)batch_instance_normalization_503_56542087)batch_instance_normalization_503_56542089)batch_instance_normalization_503_56542091*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56542086
LeakyRelu_4	LeakyReluAbatch_instance_normalization_503/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_612/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_4:activations:0conv2d_612_56542115*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56542114ī
8batch_instance_normalization_504/StatefulPartitionedCallStatefulPartitionedCall+conv2d_612/StatefulPartitionedCall:output:0)batch_instance_normalization_504_56542159)batch_instance_normalization_504_56542161)batch_instance_normalization_504_56542163*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56542158
LeakyRelu_5	LeakyReluAbatch_instance_normalization_504/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
"conv2d_613/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_5:activations:0conv2d_613_56542187*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56542186ī
8batch_instance_normalization_505/StatefulPartitionedCallStatefulPartitionedCall+conv2d_613/StatefulPartitionedCall:output:0)batch_instance_normalization_505_56542231)batch_instance_normalization_505_56542233)batch_instance_normalization_505_56542235*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56542230
LeakyRelu_6	LeakyReluAbatch_instance_normalization_505/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ī
,conv2d_transpose_100/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_6:activations:0conv2d_transpose_100_56542239*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56541751[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
concatenate_1/concatConcatV2LeakyRelu_2:activations:05conv2d_transpose_100/StatefulPartitionedCall:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_614/StatefulPartitionedCallStatefulPartitionedCallconcatenate_1/concat:output:0conv2d_614_56542252*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56542251ī
8batch_instance_normalization_506/StatefulPartitionedCallStatefulPartitionedCall+conv2d_614/StatefulPartitionedCall:output:0)batch_instance_normalization_506_56542296)batch_instance_normalization_506_56542298)batch_instance_normalization_506_56542300*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56542295
LeakyRelu_7	LeakyReluAbatch_instance_normalization_506/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
"conv2d_615/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_7:activations:0conv2d_615_56542312*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56542311ī
8batch_instance_normalization_507/StatefulPartitionedCallStatefulPartitionedCall+conv2d_615/StatefulPartitionedCall:output:0)batch_instance_normalization_507_56542356)batch_instance_normalization_507_56542358)batch_instance_normalization_507_56542360*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56542355
LeakyRelu_8	LeakyReluAbatch_instance_normalization_507/StatefulPartitionedCall:output:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
,conv2d_transpose_101/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_8:activations:0conv2d_transpose_101_56542364*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56541790[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ú
concatenate_2/concatConcatV2LeakyRelu:activations:05conv2d_transpose_101/StatefulPartitionedCall:output:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
"conv2d_616/StatefulPartitionedCallStatefulPartitionedCallconcatenate_2/concat:output:0conv2d_616_56542377*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56542376ĩ
8batch_instance_normalization_508/StatefulPartitionedCallStatefulPartitionedCall+conv2d_616/StatefulPartitionedCall:output:0)batch_instance_normalization_508_56542421)batch_instance_normalization_508_56542423)batch_instance_normalization_508_56542425*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56542420
LeakyRelu_9	LeakyReluAbatch_instance_normalization_508/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
"conv2d_617/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_9:activations:0conv2d_617_56542437*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56542436ĩ
8batch_instance_normalization_509/StatefulPartitionedCallStatefulPartitionedCall+conv2d_617/StatefulPartitionedCall:output:0)batch_instance_normalization_509_56542481)batch_instance_normalization_509_56542483)batch_instance_normalization_509_56542485*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56542480
LeakyRelu_10	LeakyReluAbatch_instance_normalization_509/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
"conv2d_618/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_10:activations:0conv2d_618_56542497*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56542496ĩ
8batch_instance_normalization_510/StatefulPartitionedCallStatefulPartitionedCall+conv2d_618/StatefulPartitionedCall:output:0)batch_instance_normalization_510_56542541)batch_instance_normalization_510_56542543)batch_instance_normalization_510_56542545*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56542540
LeakyRelu_11	LeakyReluAbatch_instance_normalization_510/StatefulPartitionedCall:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"conv2d_619/StatefulPartitionedCallStatefulPartitionedCallLeakyRelu_11:activations:0conv2d_619_56542557*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56542556u
TanhTanh+conv2d_619/StatefulPartitionedCall:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸa
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸģ

NoOpNoOp9^batch_instance_normalization_500/StatefulPartitionedCall9^batch_instance_normalization_501/StatefulPartitionedCall9^batch_instance_normalization_502/StatefulPartitionedCall9^batch_instance_normalization_503/StatefulPartitionedCall9^batch_instance_normalization_504/StatefulPartitionedCall9^batch_instance_normalization_505/StatefulPartitionedCall9^batch_instance_normalization_506/StatefulPartitionedCall9^batch_instance_normalization_507/StatefulPartitionedCall9^batch_instance_normalization_508/StatefulPartitionedCall9^batch_instance_normalization_509/StatefulPartitionedCall9^batch_instance_normalization_510/StatefulPartitionedCall#^conv2d_606/StatefulPartitionedCall#^conv2d_607/StatefulPartitionedCall#^conv2d_608/StatefulPartitionedCall#^conv2d_609/StatefulPartitionedCall#^conv2d_610/StatefulPartitionedCall#^conv2d_611/StatefulPartitionedCall#^conv2d_612/StatefulPartitionedCall#^conv2d_613/StatefulPartitionedCall#^conv2d_614/StatefulPartitionedCall#^conv2d_615/StatefulPartitionedCall#^conv2d_616/StatefulPartitionedCall#^conv2d_617/StatefulPartitionedCall#^conv2d_618/StatefulPartitionedCall#^conv2d_619/StatefulPartitionedCall-^conv2d_transpose_100/StatefulPartitionedCall-^conv2d_transpose_101/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8batch_instance_normalization_500/StatefulPartitionedCall8batch_instance_normalization_500/StatefulPartitionedCall2t
8batch_instance_normalization_501/StatefulPartitionedCall8batch_instance_normalization_501/StatefulPartitionedCall2t
8batch_instance_normalization_502/StatefulPartitionedCall8batch_instance_normalization_502/StatefulPartitionedCall2t
8batch_instance_normalization_503/StatefulPartitionedCall8batch_instance_normalization_503/StatefulPartitionedCall2t
8batch_instance_normalization_504/StatefulPartitionedCall8batch_instance_normalization_504/StatefulPartitionedCall2t
8batch_instance_normalization_505/StatefulPartitionedCall8batch_instance_normalization_505/StatefulPartitionedCall2t
8batch_instance_normalization_506/StatefulPartitionedCall8batch_instance_normalization_506/StatefulPartitionedCall2t
8batch_instance_normalization_507/StatefulPartitionedCall8batch_instance_normalization_507/StatefulPartitionedCall2t
8batch_instance_normalization_508/StatefulPartitionedCall8batch_instance_normalization_508/StatefulPartitionedCall2t
8batch_instance_normalization_509/StatefulPartitionedCall8batch_instance_normalization_509/StatefulPartitionedCall2t
8batch_instance_normalization_510/StatefulPartitionedCall8batch_instance_normalization_510/StatefulPartitionedCall2H
"conv2d_606/StatefulPartitionedCall"conv2d_606/StatefulPartitionedCall2H
"conv2d_607/StatefulPartitionedCall"conv2d_607/StatefulPartitionedCall2H
"conv2d_608/StatefulPartitionedCall"conv2d_608/StatefulPartitionedCall2H
"conv2d_609/StatefulPartitionedCall"conv2d_609/StatefulPartitionedCall2H
"conv2d_610/StatefulPartitionedCall"conv2d_610/StatefulPartitionedCall2H
"conv2d_611/StatefulPartitionedCall"conv2d_611/StatefulPartitionedCall2H
"conv2d_612/StatefulPartitionedCall"conv2d_612/StatefulPartitionedCall2H
"conv2d_613/StatefulPartitionedCall"conv2d_613/StatefulPartitionedCall2H
"conv2d_614/StatefulPartitionedCall"conv2d_614/StatefulPartitionedCall2H
"conv2d_615/StatefulPartitionedCall"conv2d_615/StatefulPartitionedCall2H
"conv2d_616/StatefulPartitionedCall"conv2d_616/StatefulPartitionedCall2H
"conv2d_617/StatefulPartitionedCall"conv2d_617/StatefulPartitionedCall2H
"conv2d_618/StatefulPartitionedCall"conv2d_618/StatefulPartitionedCall2H
"conv2d_619/StatefulPartitionedCall"conv2d_619/StatefulPartitionedCall2\
,conv2d_transpose_100/StatefulPartitionedCall,conv2d_transpose_100/StatefulPartitionedCall2\
,conv2d_transpose_101/StatefulPartitionedCall,conv2d_transpose_101/StatefulPartitionedCall:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs:YU
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56545406
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ  w
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
:ĸĸĸĸĸĸĸĸĸ  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  e
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
:ĸĸĸĸĸĸĸĸĸ  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
Š
đ
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56545843

inputs8
conv2d_readvariableop_resource:@@
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
ĩ
Ã
C__inference_batch_instance_normalization_505_layer_call_fn_56545520
x
unknown:	
	unknown_0:	
	unknown_1:	
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56542230x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
ĩ
Ã
C__inference_batch_instance_normalization_502_layer_call_fn_56545289
x
unknown:	
	unknown_0:	
	unknown_1:	
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56542014x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
Š
đ
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56541824

inputs8
conv2d_readvariableop_resource:@@
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
Ó

-__inference_conv2d_610_layer_call_fn_56545259

inputs#
unknown:
identityĒStatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56541970x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs

k
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56541714

inputs
identityĒ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ:r n
J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
 
=
#__inference__wrapped_model_56541693
input_1
input_2M
3face_g_18_conv2d_606_conv2d_readvariableop_resource:@M
3face_g_18_conv2d_607_conv2d_readvariableop_resource:@@N
3face_g_18_conv2d_608_conv2d_readvariableop_resource:@Q
Bface_g_18_batch_instance_normalization_500_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_500_mul_4_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_500_add_3_readvariableop_resource:	O
3face_g_18_conv2d_609_conv2d_readvariableop_resource:Q
Bface_g_18_batch_instance_normalization_501_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_501_mul_4_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_501_add_3_readvariableop_resource:	O
3face_g_18_conv2d_610_conv2d_readvariableop_resource:Q
Bface_g_18_batch_instance_normalization_502_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_502_mul_4_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_502_add_3_readvariableop_resource:	O
3face_g_18_conv2d_611_conv2d_readvariableop_resource:Q
Bface_g_18_batch_instance_normalization_503_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_503_mul_4_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_503_add_3_readvariableop_resource:	O
3face_g_18_conv2d_612_conv2d_readvariableop_resource:Q
Bface_g_18_batch_instance_normalization_504_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_504_mul_4_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_504_add_3_readvariableop_resource:	O
3face_g_18_conv2d_613_conv2d_readvariableop_resource:Q
Bface_g_18_batch_instance_normalization_505_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_505_mul_4_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_505_add_3_readvariableop_resource:	c
Gface_g_18_conv2d_transpose_100_conv2d_transpose_readvariableop_resource:O
3face_g_18_conv2d_614_conv2d_readvariableop_resource:Q
Bface_g_18_batch_instance_normalization_506_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_506_mul_4_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_506_add_3_readvariableop_resource:	O
3face_g_18_conv2d_615_conv2d_readvariableop_resource:Q
Bface_g_18_batch_instance_normalization_507_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_507_mul_4_readvariableop_resource:	W
Hface_g_18_batch_instance_normalization_507_add_3_readvariableop_resource:	b
Gface_g_18_conv2d_transpose_101_conv2d_transpose_readvariableop_resource:@N
3face_g_18_conv2d_616_conv2d_readvariableop_resource:@P
Bface_g_18_batch_instance_normalization_508_readvariableop_resource:@V
Hface_g_18_batch_instance_normalization_508_mul_4_readvariableop_resource:@V
Hface_g_18_batch_instance_normalization_508_add_3_readvariableop_resource:@M
3face_g_18_conv2d_617_conv2d_readvariableop_resource:@@P
Bface_g_18_batch_instance_normalization_509_readvariableop_resource:@V
Hface_g_18_batch_instance_normalization_509_mul_4_readvariableop_resource:@V
Hface_g_18_batch_instance_normalization_509_add_3_readvariableop_resource:@M
3face_g_18_conv2d_618_conv2d_readvariableop_resource:@P
Bface_g_18_batch_instance_normalization_510_readvariableop_resource:V
Hface_g_18_batch_instance_normalization_510_mul_4_readvariableop_resource:V
Hface_g_18_batch_instance_normalization_510_add_3_readvariableop_resource:M
3face_g_18_conv2d_619_conv2d_readvariableop_resource:
identityĒ9face_g_18/batch_instance_normalization_500/ReadVariableOpĒ;face_g_18/batch_instance_normalization_500/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_500/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_500/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_501/ReadVariableOpĒ;face_g_18/batch_instance_normalization_501/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_501/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_501/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_502/ReadVariableOpĒ;face_g_18/batch_instance_normalization_502/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_502/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_502/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_503/ReadVariableOpĒ;face_g_18/batch_instance_normalization_503/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_503/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_503/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_504/ReadVariableOpĒ;face_g_18/batch_instance_normalization_504/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_504/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_504/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_505/ReadVariableOpĒ;face_g_18/batch_instance_normalization_505/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_505/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_505/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_506/ReadVariableOpĒ;face_g_18/batch_instance_normalization_506/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_506/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_506/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_507/ReadVariableOpĒ;face_g_18/batch_instance_normalization_507/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_507/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_507/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_508/ReadVariableOpĒ;face_g_18/batch_instance_normalization_508/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_508/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_508/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_509/ReadVariableOpĒ;face_g_18/batch_instance_normalization_509/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_509/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_509/mul_4/ReadVariableOpĒ9face_g_18/batch_instance_normalization_510/ReadVariableOpĒ;face_g_18/batch_instance_normalization_510/ReadVariableOp_1Ē?face_g_18/batch_instance_normalization_510/add_3/ReadVariableOpĒ?face_g_18/batch_instance_normalization_510/mul_4/ReadVariableOpĒ*face_g_18/conv2d_606/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_607/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_608/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_609/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_610/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_611/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_612/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_613/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_614/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_615/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_616/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_617/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_618/Conv2D/ReadVariableOpĒ*face_g_18/conv2d_619/Conv2D/ReadVariableOpĒ>face_g_18/conv2d_transpose_100/conv2d_transpose/ReadVariableOpĒ>face_g_18/conv2d_transpose_101/conv2d_transpose/ReadVariableOpc
!face_g_18/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ŧ
face_g_18/concatenate/concatConcatV2input_1input_2*face_g_18/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸĶ
*face_g_18/conv2d_606/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_606_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ä
face_g_18/conv2d_606/Conv2DConv2D%face_g_18/concatenate/concat:output:02face_g_18/conv2d_606/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
Ķ
*face_g_18/conv2d_607/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_607_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ã
face_g_18/conv2d_607/Conv2DConv2D$face_g_18/conv2d_606/Conv2D:output:02face_g_18/conv2d_607/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
y
face_g_18/LeakyRelu	LeakyRelu$face_g_18/conv2d_607/Conv2D:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ū
#face_g_18/max_pooling2d_100/MaxPoolMaxPool!face_g_18/LeakyRelu:activations:0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@*
ksize
*
paddingVALID*
strides
§
*face_g_18/conv2d_608/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_608_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0ę
face_g_18/conv2d_608/Conv2DConv2D,face_g_18/max_pooling2d_100/MaxPool:output:02face_g_18/conv2d_608/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

Iface_g_18/batch_instance_normalization_500/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ü
7face_g_18/batch_instance_normalization_500/moments/meanMean$face_g_18/conv2d_608/Conv2D:output:0Rface_g_18/batch_instance_normalization_500/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ã
?face_g_18/batch_instance_normalization_500/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_500/moments/mean:output:0*
T0*'
_output_shapes
:
Dface_g_18/batch_instance_normalization_500/moments/SquaredDifferenceSquaredDifference$face_g_18/conv2d_608/Conv2D:output:0Hface_g_18/batch_instance_normalization_500/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ē
Mface_g_18/batch_instance_normalization_500/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ļ
;face_g_18/batch_instance_normalization_500/moments/varianceMeanHface_g_18/batch_instance_normalization_500/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_500/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ø
.face_g_18/batch_instance_normalization_500/subSub$face_g_18/conv2d_608/Conv2D:output:0@face_g_18/batch_instance_normalization_500/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
0face_g_18/batch_instance_normalization_500/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ę
.face_g_18/batch_instance_normalization_500/addAddV2Dface_g_18/batch_instance_normalization_500/moments/variance:output:09face_g_18/batch_instance_normalization_500/add/y:output:0*
T0*'
_output_shapes
:
0face_g_18/batch_instance_normalization_500/RsqrtRsqrt2face_g_18/batch_instance_normalization_500/add:z:0*
T0*'
_output_shapes
:Ú
.face_g_18/batch_instance_normalization_500/mulMul2face_g_18/batch_instance_normalization_500/sub:z:04face_g_18/batch_instance_normalization_500/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Kface_g_18/batch_instance_normalization_500/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_500/moments_1/meanMean$face_g_18/conv2d_608/Conv2D:output:0Tface_g_18/batch_instance_normalization_500/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ð
Aface_g_18/batch_instance_normalization_500/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_500/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_500/moments_1/SquaredDifferenceSquaredDifference$face_g_18/conv2d_608/Conv2D:output:0Jface_g_18/batch_instance_normalization_500/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ 
Oface_g_18/batch_instance_normalization_500/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ·
=face_g_18/batch_instance_normalization_500/moments_1/varianceMeanJface_g_18/batch_instance_normalization_500/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_500/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ü
0face_g_18/batch_instance_normalization_500/sub_1Sub$face_g_18/conv2d_608/Conv2D:output:0Bface_g_18/batch_instance_normalization_500/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@w
2face_g_18/batch_instance_normalization_500/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ų
0face_g_18/batch_instance_normalization_500/add_1AddV2Fface_g_18/batch_instance_normalization_500/moments_1/variance:output:0;face_g_18/batch_instance_normalization_500/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸŽ
2face_g_18/batch_instance_normalization_500/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_500/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸā
0face_g_18/batch_instance_normalization_500/mul_1Mul4face_g_18/batch_instance_normalization_500/sub_1:z:06face_g_18/batch_instance_normalization_500/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@đ
9face_g_18/batch_instance_normalization_500/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_500_readvariableop_resource*
_output_shapes	
:*
dtype0é
0face_g_18/batch_instance_normalization_500/mul_2MulAface_g_18/batch_instance_normalization_500/ReadVariableOp:value:02face_g_18/batch_instance_normalization_500/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ŧ
;face_g_18/batch_instance_normalization_500/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_500_readvariableop_resource*
_output_shapes	
:*
dtype0w
2face_g_18/batch_instance_normalization_500/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ß
0face_g_18/batch_instance_normalization_500/sub_2Sub;face_g_18/batch_instance_normalization_500/sub_2/x:output:0Cface_g_18/batch_instance_normalization_500/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Þ
0face_g_18/batch_instance_normalization_500/mul_3Mul4face_g_18/batch_instance_normalization_500/sub_2:z:04face_g_18/batch_instance_normalization_500/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ā
0face_g_18/batch_instance_normalization_500/add_2AddV24face_g_18/batch_instance_normalization_500/mul_2:z:04face_g_18/batch_instance_normalization_500/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Å
?face_g_18/batch_instance_normalization_500/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_500_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0ņ
0face_g_18/batch_instance_normalization_500/mul_4Mul4face_g_18/batch_instance_normalization_500/add_2:z:0Gface_g_18/batch_instance_normalization_500/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Å
?face_g_18/batch_instance_normalization_500/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_500_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0ó
0face_g_18/batch_instance_normalization_500/add_3AddV24face_g_18/batch_instance_normalization_500/mul_4:z:0Gface_g_18/batch_instance_normalization_500/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
face_g_18/LeakyRelu_1	LeakyRelu4face_g_18/batch_instance_normalization_500/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ļ
*face_g_18/conv2d_609/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_609_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0á
face_g_18/conv2d_609/Conv2DConv2D#face_g_18/LeakyRelu_1:activations:02face_g_18/conv2d_609/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

Iface_g_18/batch_instance_normalization_501/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ü
7face_g_18/batch_instance_normalization_501/moments/meanMean$face_g_18/conv2d_609/Conv2D:output:0Rface_g_18/batch_instance_normalization_501/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ã
?face_g_18/batch_instance_normalization_501/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_501/moments/mean:output:0*
T0*'
_output_shapes
:
Dface_g_18/batch_instance_normalization_501/moments/SquaredDifferenceSquaredDifference$face_g_18/conv2d_609/Conv2D:output:0Hface_g_18/batch_instance_normalization_501/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ē
Mface_g_18/batch_instance_normalization_501/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ļ
;face_g_18/batch_instance_normalization_501/moments/varianceMeanHface_g_18/batch_instance_normalization_501/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_501/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ø
.face_g_18/batch_instance_normalization_501/subSub$face_g_18/conv2d_609/Conv2D:output:0@face_g_18/batch_instance_normalization_501/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
0face_g_18/batch_instance_normalization_501/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ę
.face_g_18/batch_instance_normalization_501/addAddV2Dface_g_18/batch_instance_normalization_501/moments/variance:output:09face_g_18/batch_instance_normalization_501/add/y:output:0*
T0*'
_output_shapes
:
0face_g_18/batch_instance_normalization_501/RsqrtRsqrt2face_g_18/batch_instance_normalization_501/add:z:0*
T0*'
_output_shapes
:Ú
.face_g_18/batch_instance_normalization_501/mulMul2face_g_18/batch_instance_normalization_501/sub:z:04face_g_18/batch_instance_normalization_501/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Kface_g_18/batch_instance_normalization_501/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_501/moments_1/meanMean$face_g_18/conv2d_609/Conv2D:output:0Tface_g_18/batch_instance_normalization_501/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ð
Aface_g_18/batch_instance_normalization_501/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_501/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_501/moments_1/SquaredDifferenceSquaredDifference$face_g_18/conv2d_609/Conv2D:output:0Jface_g_18/batch_instance_normalization_501/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ 
Oface_g_18/batch_instance_normalization_501/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ·
=face_g_18/batch_instance_normalization_501/moments_1/varianceMeanJface_g_18/batch_instance_normalization_501/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_501/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ü
0face_g_18/batch_instance_normalization_501/sub_1Sub$face_g_18/conv2d_609/Conv2D:output:0Bface_g_18/batch_instance_normalization_501/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@w
2face_g_18/batch_instance_normalization_501/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ų
0face_g_18/batch_instance_normalization_501/add_1AddV2Fface_g_18/batch_instance_normalization_501/moments_1/variance:output:0;face_g_18/batch_instance_normalization_501/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸŽ
2face_g_18/batch_instance_normalization_501/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_501/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸā
0face_g_18/batch_instance_normalization_501/mul_1Mul4face_g_18/batch_instance_normalization_501/sub_1:z:06face_g_18/batch_instance_normalization_501/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@đ
9face_g_18/batch_instance_normalization_501/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_501_readvariableop_resource*
_output_shapes	
:*
dtype0é
0face_g_18/batch_instance_normalization_501/mul_2MulAface_g_18/batch_instance_normalization_501/ReadVariableOp:value:02face_g_18/batch_instance_normalization_501/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ŧ
;face_g_18/batch_instance_normalization_501/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_501_readvariableop_resource*
_output_shapes	
:*
dtype0w
2face_g_18/batch_instance_normalization_501/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ß
0face_g_18/batch_instance_normalization_501/sub_2Sub;face_g_18/batch_instance_normalization_501/sub_2/x:output:0Cface_g_18/batch_instance_normalization_501/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Þ
0face_g_18/batch_instance_normalization_501/mul_3Mul4face_g_18/batch_instance_normalization_501/sub_2:z:04face_g_18/batch_instance_normalization_501/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ā
0face_g_18/batch_instance_normalization_501/add_2AddV24face_g_18/batch_instance_normalization_501/mul_2:z:04face_g_18/batch_instance_normalization_501/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Å
?face_g_18/batch_instance_normalization_501/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_501_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0ņ
0face_g_18/batch_instance_normalization_501/mul_4Mul4face_g_18/batch_instance_normalization_501/add_2:z:0Gface_g_18/batch_instance_normalization_501/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Å
?face_g_18/batch_instance_normalization_501/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_501_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0ó
0face_g_18/batch_instance_normalization_501/add_3AddV24face_g_18/batch_instance_normalization_501/mul_4:z:0Gface_g_18/batch_instance_normalization_501/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
face_g_18/LeakyRelu_2	LeakyRelu4face_g_18/batch_instance_normalization_501/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Á
#face_g_18/max_pooling2d_101/MaxPoolMaxPool#face_g_18/LeakyRelu_2:activations:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *
ksize
*
paddingVALID*
strides
z
)face_g_18/conv2d_610/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
(face_g_18/conv2d_610/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
!face_g_18/conv2d_610/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
Hface_g_18/conv2d_610/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Ķ
Eface_g_18/conv2d_610/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            Ģ
Bface_g_18/conv2d_610/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                
6face_g_18/conv2d_610/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
3face_g_18/conv2d_610/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            Ī
*face_g_18/conv2d_610/Conv2D/SpaceToBatchNDSpaceToBatchND,face_g_18/max_pooling2d_101/MaxPool:output:0?face_g_18/conv2d_610/Conv2D/SpaceToBatchND/block_shape:output:0<face_g_18/conv2d_610/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸĻ
*face_g_18/conv2d_610/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_610_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ō
face_g_18/conv2d_610/Conv2DConv2D3face_g_18/conv2d_610/Conv2D/SpaceToBatchND:output:02face_g_18/conv2d_610/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides

6face_g_18/conv2d_610/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
0face_g_18/conv2d_610/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                
*face_g_18/conv2d_610/Conv2D/BatchToSpaceNDBatchToSpaceND$face_g_18/conv2d_610/Conv2D:output:0?face_g_18/conv2d_610/Conv2D/BatchToSpaceND/block_shape:output:09face_g_18/conv2d_610/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Iface_g_18/batch_instance_normalization_502/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
7face_g_18/batch_instance_normalization_502/moments/meanMean3face_g_18/conv2d_610/Conv2D/BatchToSpaceND:output:0Rface_g_18/batch_instance_normalization_502/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ã
?face_g_18/batch_instance_normalization_502/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_502/moments/mean:output:0*
T0*'
_output_shapes
:
Dface_g_18/batch_instance_normalization_502/moments/SquaredDifferenceSquaredDifference3face_g_18/conv2d_610/Conv2D/BatchToSpaceND:output:0Hface_g_18/batch_instance_normalization_502/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ē
Mface_g_18/batch_instance_normalization_502/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ļ
;face_g_18/batch_instance_normalization_502/moments/varianceMeanHface_g_18/batch_instance_normalization_502/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_502/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(į
.face_g_18/batch_instance_normalization_502/subSub3face_g_18/conv2d_610/Conv2D/BatchToSpaceND:output:0@face_g_18/batch_instance_normalization_502/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
0face_g_18/batch_instance_normalization_502/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ę
.face_g_18/batch_instance_normalization_502/addAddV2Dface_g_18/batch_instance_normalization_502/moments/variance:output:09face_g_18/batch_instance_normalization_502/add/y:output:0*
T0*'
_output_shapes
:
0face_g_18/batch_instance_normalization_502/RsqrtRsqrt2face_g_18/batch_instance_normalization_502/add:z:0*
T0*'
_output_shapes
:Ú
.face_g_18/batch_instance_normalization_502/mulMul2face_g_18/batch_instance_normalization_502/sub:z:04face_g_18/batch_instance_normalization_502/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Kface_g_18/batch_instance_normalization_502/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_502/moments_1/meanMean3face_g_18/conv2d_610/Conv2D/BatchToSpaceND:output:0Tface_g_18/batch_instance_normalization_502/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ð
Aface_g_18/batch_instance_normalization_502/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_502/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_502/moments_1/SquaredDifferenceSquaredDifference3face_g_18/conv2d_610/Conv2D/BatchToSpaceND:output:0Jface_g_18/batch_instance_normalization_502/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ   
Oface_g_18/batch_instance_normalization_502/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ·
=face_g_18/batch_instance_normalization_502/moments_1/varianceMeanJface_g_18/batch_instance_normalization_502/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_502/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ë
0face_g_18/batch_instance_normalization_502/sub_1Sub3face_g_18/conv2d_610/Conv2D/BatchToSpaceND:output:0Bface_g_18/batch_instance_normalization_502/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  w
2face_g_18/batch_instance_normalization_502/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ų
0face_g_18/batch_instance_normalization_502/add_1AddV2Fface_g_18/batch_instance_normalization_502/moments_1/variance:output:0;face_g_18/batch_instance_normalization_502/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸŽ
2face_g_18/batch_instance_normalization_502/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_502/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸā
0face_g_18/batch_instance_normalization_502/mul_1Mul4face_g_18/batch_instance_normalization_502/sub_1:z:06face_g_18/batch_instance_normalization_502/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  đ
9face_g_18/batch_instance_normalization_502/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_502_readvariableop_resource*
_output_shapes	
:*
dtype0é
0face_g_18/batch_instance_normalization_502/mul_2MulAface_g_18/batch_instance_normalization_502/ReadVariableOp:value:02face_g_18/batch_instance_normalization_502/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ŧ
;face_g_18/batch_instance_normalization_502/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_502_readvariableop_resource*
_output_shapes	
:*
dtype0w
2face_g_18/batch_instance_normalization_502/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ß
0face_g_18/batch_instance_normalization_502/sub_2Sub;face_g_18/batch_instance_normalization_502/sub_2/x:output:0Cface_g_18/batch_instance_normalization_502/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Þ
0face_g_18/batch_instance_normalization_502/mul_3Mul4face_g_18/batch_instance_normalization_502/sub_2:z:04face_g_18/batch_instance_normalization_502/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ā
0face_g_18/batch_instance_normalization_502/add_2AddV24face_g_18/batch_instance_normalization_502/mul_2:z:04face_g_18/batch_instance_normalization_502/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Å
?face_g_18/batch_instance_normalization_502/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_502_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0ņ
0face_g_18/batch_instance_normalization_502/mul_4Mul4face_g_18/batch_instance_normalization_502/add_2:z:0Gface_g_18/batch_instance_normalization_502/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Å
?face_g_18/batch_instance_normalization_502/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_502_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0ó
0face_g_18/batch_instance_normalization_502/add_3AddV24face_g_18/batch_instance_normalization_502/mul_4:z:0Gface_g_18/batch_instance_normalization_502/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
face_g_18/LeakyRelu_3	LeakyRelu4face_g_18/batch_instance_normalization_502/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  z
)face_g_18/conv2d_611/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
(face_g_18/conv2d_611/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
!face_g_18/conv2d_611/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
Hface_g_18/conv2d_611/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Ķ
Eface_g_18/conv2d_611/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            Ģ
Bface_g_18/conv2d_611/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                
6face_g_18/conv2d_611/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
3face_g_18/conv2d_611/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
*face_g_18/conv2d_611/Conv2D/SpaceToBatchNDSpaceToBatchND#face_g_18/LeakyRelu_3:activations:0?face_g_18/conv2d_611/Conv2D/SpaceToBatchND/block_shape:output:0<face_g_18/conv2d_611/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ

Ļ
*face_g_18/conv2d_611/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_611_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ō
face_g_18/conv2d_611/Conv2DConv2D3face_g_18/conv2d_611/Conv2D/SpaceToBatchND:output:02face_g_18/conv2d_611/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides

6face_g_18/conv2d_611/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
0face_g_18/conv2d_611/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                
*face_g_18/conv2d_611/Conv2D/BatchToSpaceNDBatchToSpaceND$face_g_18/conv2d_611/Conv2D:output:0?face_g_18/conv2d_611/Conv2D/BatchToSpaceND/block_shape:output:09face_g_18/conv2d_611/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Iface_g_18/batch_instance_normalization_503/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
7face_g_18/batch_instance_normalization_503/moments/meanMean3face_g_18/conv2d_611/Conv2D/BatchToSpaceND:output:0Rface_g_18/batch_instance_normalization_503/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ã
?face_g_18/batch_instance_normalization_503/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_503/moments/mean:output:0*
T0*'
_output_shapes
:
Dface_g_18/batch_instance_normalization_503/moments/SquaredDifferenceSquaredDifference3face_g_18/conv2d_611/Conv2D/BatchToSpaceND:output:0Hface_g_18/batch_instance_normalization_503/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ē
Mface_g_18/batch_instance_normalization_503/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ļ
;face_g_18/batch_instance_normalization_503/moments/varianceMeanHface_g_18/batch_instance_normalization_503/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_503/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(į
.face_g_18/batch_instance_normalization_503/subSub3face_g_18/conv2d_611/Conv2D/BatchToSpaceND:output:0@face_g_18/batch_instance_normalization_503/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
0face_g_18/batch_instance_normalization_503/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ę
.face_g_18/batch_instance_normalization_503/addAddV2Dface_g_18/batch_instance_normalization_503/moments/variance:output:09face_g_18/batch_instance_normalization_503/add/y:output:0*
T0*'
_output_shapes
:
0face_g_18/batch_instance_normalization_503/RsqrtRsqrt2face_g_18/batch_instance_normalization_503/add:z:0*
T0*'
_output_shapes
:Ú
.face_g_18/batch_instance_normalization_503/mulMul2face_g_18/batch_instance_normalization_503/sub:z:04face_g_18/batch_instance_normalization_503/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Kface_g_18/batch_instance_normalization_503/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_503/moments_1/meanMean3face_g_18/conv2d_611/Conv2D/BatchToSpaceND:output:0Tface_g_18/batch_instance_normalization_503/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ð
Aface_g_18/batch_instance_normalization_503/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_503/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_503/moments_1/SquaredDifferenceSquaredDifference3face_g_18/conv2d_611/Conv2D/BatchToSpaceND:output:0Jface_g_18/batch_instance_normalization_503/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ   
Oface_g_18/batch_instance_normalization_503/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ·
=face_g_18/batch_instance_normalization_503/moments_1/varianceMeanJface_g_18/batch_instance_normalization_503/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_503/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ë
0face_g_18/batch_instance_normalization_503/sub_1Sub3face_g_18/conv2d_611/Conv2D/BatchToSpaceND:output:0Bface_g_18/batch_instance_normalization_503/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  w
2face_g_18/batch_instance_normalization_503/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ų
0face_g_18/batch_instance_normalization_503/add_1AddV2Fface_g_18/batch_instance_normalization_503/moments_1/variance:output:0;face_g_18/batch_instance_normalization_503/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸŽ
2face_g_18/batch_instance_normalization_503/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_503/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸā
0face_g_18/batch_instance_normalization_503/mul_1Mul4face_g_18/batch_instance_normalization_503/sub_1:z:06face_g_18/batch_instance_normalization_503/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  đ
9face_g_18/batch_instance_normalization_503/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_503_readvariableop_resource*
_output_shapes	
:*
dtype0é
0face_g_18/batch_instance_normalization_503/mul_2MulAface_g_18/batch_instance_normalization_503/ReadVariableOp:value:02face_g_18/batch_instance_normalization_503/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ŧ
;face_g_18/batch_instance_normalization_503/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_503_readvariableop_resource*
_output_shapes	
:*
dtype0w
2face_g_18/batch_instance_normalization_503/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ß
0face_g_18/batch_instance_normalization_503/sub_2Sub;face_g_18/batch_instance_normalization_503/sub_2/x:output:0Cface_g_18/batch_instance_normalization_503/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Þ
0face_g_18/batch_instance_normalization_503/mul_3Mul4face_g_18/batch_instance_normalization_503/sub_2:z:04face_g_18/batch_instance_normalization_503/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ā
0face_g_18/batch_instance_normalization_503/add_2AddV24face_g_18/batch_instance_normalization_503/mul_2:z:04face_g_18/batch_instance_normalization_503/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Å
?face_g_18/batch_instance_normalization_503/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_503_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0ņ
0face_g_18/batch_instance_normalization_503/mul_4Mul4face_g_18/batch_instance_normalization_503/add_2:z:0Gface_g_18/batch_instance_normalization_503/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Å
?face_g_18/batch_instance_normalization_503/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_503_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0ó
0face_g_18/batch_instance_normalization_503/add_3AddV24face_g_18/batch_instance_normalization_503/mul_4:z:0Gface_g_18/batch_instance_normalization_503/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
face_g_18/LeakyRelu_4	LeakyRelu4face_g_18/batch_instance_normalization_503/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  z
)face_g_18/conv2d_612/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
(face_g_18/conv2d_612/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
!face_g_18/conv2d_612/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
Hface_g_18/conv2d_612/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Ķ
Eface_g_18/conv2d_612/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            Ģ
Bface_g_18/conv2d_612/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                
6face_g_18/conv2d_612/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
3face_g_18/conv2d_612/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
*face_g_18/conv2d_612/Conv2D/SpaceToBatchNDSpaceToBatchND#face_g_18/LeakyRelu_4:activations:0?face_g_18/conv2d_612/Conv2D/SpaceToBatchND/block_shape:output:0<face_g_18/conv2d_612/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸĻ
*face_g_18/conv2d_612/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_612_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ō
face_g_18/conv2d_612/Conv2DConv2D3face_g_18/conv2d_612/Conv2D/SpaceToBatchND:output:02face_g_18/conv2d_612/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides

6face_g_18/conv2d_612/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
0face_g_18/conv2d_612/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                
*face_g_18/conv2d_612/Conv2D/BatchToSpaceNDBatchToSpaceND$face_g_18/conv2d_612/Conv2D:output:0?face_g_18/conv2d_612/Conv2D/BatchToSpaceND/block_shape:output:09face_g_18/conv2d_612/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Iface_g_18/batch_instance_normalization_504/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
7face_g_18/batch_instance_normalization_504/moments/meanMean3face_g_18/conv2d_612/Conv2D/BatchToSpaceND:output:0Rface_g_18/batch_instance_normalization_504/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ã
?face_g_18/batch_instance_normalization_504/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_504/moments/mean:output:0*
T0*'
_output_shapes
:
Dface_g_18/batch_instance_normalization_504/moments/SquaredDifferenceSquaredDifference3face_g_18/conv2d_612/Conv2D/BatchToSpaceND:output:0Hface_g_18/batch_instance_normalization_504/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ē
Mface_g_18/batch_instance_normalization_504/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ļ
;face_g_18/batch_instance_normalization_504/moments/varianceMeanHface_g_18/batch_instance_normalization_504/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_504/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(į
.face_g_18/batch_instance_normalization_504/subSub3face_g_18/conv2d_612/Conv2D/BatchToSpaceND:output:0@face_g_18/batch_instance_normalization_504/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
0face_g_18/batch_instance_normalization_504/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ę
.face_g_18/batch_instance_normalization_504/addAddV2Dface_g_18/batch_instance_normalization_504/moments/variance:output:09face_g_18/batch_instance_normalization_504/add/y:output:0*
T0*'
_output_shapes
:
0face_g_18/batch_instance_normalization_504/RsqrtRsqrt2face_g_18/batch_instance_normalization_504/add:z:0*
T0*'
_output_shapes
:Ú
.face_g_18/batch_instance_normalization_504/mulMul2face_g_18/batch_instance_normalization_504/sub:z:04face_g_18/batch_instance_normalization_504/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Kface_g_18/batch_instance_normalization_504/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_504/moments_1/meanMean3face_g_18/conv2d_612/Conv2D/BatchToSpaceND:output:0Tface_g_18/batch_instance_normalization_504/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ð
Aface_g_18/batch_instance_normalization_504/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_504/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_504/moments_1/SquaredDifferenceSquaredDifference3face_g_18/conv2d_612/Conv2D/BatchToSpaceND:output:0Jface_g_18/batch_instance_normalization_504/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ   
Oface_g_18/batch_instance_normalization_504/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ·
=face_g_18/batch_instance_normalization_504/moments_1/varianceMeanJface_g_18/batch_instance_normalization_504/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_504/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ë
0face_g_18/batch_instance_normalization_504/sub_1Sub3face_g_18/conv2d_612/Conv2D/BatchToSpaceND:output:0Bface_g_18/batch_instance_normalization_504/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  w
2face_g_18/batch_instance_normalization_504/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ų
0face_g_18/batch_instance_normalization_504/add_1AddV2Fface_g_18/batch_instance_normalization_504/moments_1/variance:output:0;face_g_18/batch_instance_normalization_504/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸŽ
2face_g_18/batch_instance_normalization_504/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_504/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸā
0face_g_18/batch_instance_normalization_504/mul_1Mul4face_g_18/batch_instance_normalization_504/sub_1:z:06face_g_18/batch_instance_normalization_504/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  đ
9face_g_18/batch_instance_normalization_504/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_504_readvariableop_resource*
_output_shapes	
:*
dtype0é
0face_g_18/batch_instance_normalization_504/mul_2MulAface_g_18/batch_instance_normalization_504/ReadVariableOp:value:02face_g_18/batch_instance_normalization_504/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ŧ
;face_g_18/batch_instance_normalization_504/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_504_readvariableop_resource*
_output_shapes	
:*
dtype0w
2face_g_18/batch_instance_normalization_504/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ß
0face_g_18/batch_instance_normalization_504/sub_2Sub;face_g_18/batch_instance_normalization_504/sub_2/x:output:0Cface_g_18/batch_instance_normalization_504/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Þ
0face_g_18/batch_instance_normalization_504/mul_3Mul4face_g_18/batch_instance_normalization_504/sub_2:z:04face_g_18/batch_instance_normalization_504/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ā
0face_g_18/batch_instance_normalization_504/add_2AddV24face_g_18/batch_instance_normalization_504/mul_2:z:04face_g_18/batch_instance_normalization_504/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Å
?face_g_18/batch_instance_normalization_504/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_504_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0ņ
0face_g_18/batch_instance_normalization_504/mul_4Mul4face_g_18/batch_instance_normalization_504/add_2:z:0Gface_g_18/batch_instance_normalization_504/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Å
?face_g_18/batch_instance_normalization_504/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_504_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0ó
0face_g_18/batch_instance_normalization_504/add_3AddV24face_g_18/batch_instance_normalization_504/mul_4:z:0Gface_g_18/batch_instance_normalization_504/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
face_g_18/LeakyRelu_5	LeakyRelu4face_g_18/batch_instance_normalization_504/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  z
)face_g_18/conv2d_613/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      
(face_g_18/conv2d_613/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            
!face_g_18/conv2d_613/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
Hface_g_18/conv2d_613/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        Ķ
Eface_g_18/conv2d_613/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            Ģ
Bface_g_18/conv2d_613/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                
6face_g_18/conv2d_613/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
3face_g_18/conv2d_613/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
*face_g_18/conv2d_613/Conv2D/SpaceToBatchNDSpaceToBatchND#face_g_18/LeakyRelu_5:activations:0?face_g_18/conv2d_613/Conv2D/SpaceToBatchND/block_shape:output:0<face_g_18/conv2d_613/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸĻ
*face_g_18/conv2d_613/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_613_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ō
face_g_18/conv2d_613/Conv2DConv2D3face_g_18/conv2d_613/Conv2D/SpaceToBatchND:output:02face_g_18/conv2d_613/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides

6face_g_18/conv2d_613/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
0face_g_18/conv2d_613/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                
*face_g_18/conv2d_613/Conv2D/BatchToSpaceNDBatchToSpaceND$face_g_18/conv2d_613/Conv2D:output:0?face_g_18/conv2d_613/Conv2D/BatchToSpaceND/block_shape:output:09face_g_18/conv2d_613/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Iface_g_18/batch_instance_normalization_505/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
7face_g_18/batch_instance_normalization_505/moments/meanMean3face_g_18/conv2d_613/Conv2D/BatchToSpaceND:output:0Rface_g_18/batch_instance_normalization_505/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ã
?face_g_18/batch_instance_normalization_505/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_505/moments/mean:output:0*
T0*'
_output_shapes
:
Dface_g_18/batch_instance_normalization_505/moments/SquaredDifferenceSquaredDifference3face_g_18/conv2d_613/Conv2D/BatchToSpaceND:output:0Hface_g_18/batch_instance_normalization_505/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ē
Mface_g_18/batch_instance_normalization_505/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ļ
;face_g_18/batch_instance_normalization_505/moments/varianceMeanHface_g_18/batch_instance_normalization_505/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_505/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(į
.face_g_18/batch_instance_normalization_505/subSub3face_g_18/conv2d_613/Conv2D/BatchToSpaceND:output:0@face_g_18/batch_instance_normalization_505/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
0face_g_18/batch_instance_normalization_505/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ę
.face_g_18/batch_instance_normalization_505/addAddV2Dface_g_18/batch_instance_normalization_505/moments/variance:output:09face_g_18/batch_instance_normalization_505/add/y:output:0*
T0*'
_output_shapes
:
0face_g_18/batch_instance_normalization_505/RsqrtRsqrt2face_g_18/batch_instance_normalization_505/add:z:0*
T0*'
_output_shapes
:Ú
.face_g_18/batch_instance_normalization_505/mulMul2face_g_18/batch_instance_normalization_505/sub:z:04face_g_18/batch_instance_normalization_505/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Kface_g_18/batch_instance_normalization_505/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_505/moments_1/meanMean3face_g_18/conv2d_613/Conv2D/BatchToSpaceND:output:0Tface_g_18/batch_instance_normalization_505/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ð
Aface_g_18/batch_instance_normalization_505/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_505/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_505/moments_1/SquaredDifferenceSquaredDifference3face_g_18/conv2d_613/Conv2D/BatchToSpaceND:output:0Jface_g_18/batch_instance_normalization_505/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ   
Oface_g_18/batch_instance_normalization_505/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ·
=face_g_18/batch_instance_normalization_505/moments_1/varianceMeanJface_g_18/batch_instance_normalization_505/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_505/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ë
0face_g_18/batch_instance_normalization_505/sub_1Sub3face_g_18/conv2d_613/Conv2D/BatchToSpaceND:output:0Bface_g_18/batch_instance_normalization_505/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  w
2face_g_18/batch_instance_normalization_505/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ų
0face_g_18/batch_instance_normalization_505/add_1AddV2Fface_g_18/batch_instance_normalization_505/moments_1/variance:output:0;face_g_18/batch_instance_normalization_505/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸŽ
2face_g_18/batch_instance_normalization_505/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_505/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸā
0face_g_18/batch_instance_normalization_505/mul_1Mul4face_g_18/batch_instance_normalization_505/sub_1:z:06face_g_18/batch_instance_normalization_505/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  đ
9face_g_18/batch_instance_normalization_505/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_505_readvariableop_resource*
_output_shapes	
:*
dtype0é
0face_g_18/batch_instance_normalization_505/mul_2MulAface_g_18/batch_instance_normalization_505/ReadVariableOp:value:02face_g_18/batch_instance_normalization_505/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ŧ
;face_g_18/batch_instance_normalization_505/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_505_readvariableop_resource*
_output_shapes	
:*
dtype0w
2face_g_18/batch_instance_normalization_505/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ß
0face_g_18/batch_instance_normalization_505/sub_2Sub;face_g_18/batch_instance_normalization_505/sub_2/x:output:0Cface_g_18/batch_instance_normalization_505/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Þ
0face_g_18/batch_instance_normalization_505/mul_3Mul4face_g_18/batch_instance_normalization_505/sub_2:z:04face_g_18/batch_instance_normalization_505/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ā
0face_g_18/batch_instance_normalization_505/add_2AddV24face_g_18/batch_instance_normalization_505/mul_2:z:04face_g_18/batch_instance_normalization_505/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Å
?face_g_18/batch_instance_normalization_505/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_505_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0ņ
0face_g_18/batch_instance_normalization_505/mul_4Mul4face_g_18/batch_instance_normalization_505/add_2:z:0Gface_g_18/batch_instance_normalization_505/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Å
?face_g_18/batch_instance_normalization_505/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_505_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0ó
0face_g_18/batch_instance_normalization_505/add_3AddV24face_g_18/batch_instance_normalization_505/mul_4:z:0Gface_g_18/batch_instance_normalization_505/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
face_g_18/LeakyRelu_6	LeakyRelu4face_g_18/batch_instance_normalization_505/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  w
$face_g_18/conv2d_transpose_100/ShapeShape#face_g_18/LeakyRelu_6:activations:0*
T0*
_output_shapes
:|
2face_g_18/conv2d_transpose_100/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4face_g_18/conv2d_transpose_100/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4face_g_18/conv2d_transpose_100/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ė
,face_g_18/conv2d_transpose_100/strided_sliceStridedSlice-face_g_18/conv2d_transpose_100/Shape:output:0;face_g_18/conv2d_transpose_100/strided_slice/stack:output:0=face_g_18/conv2d_transpose_100/strided_slice/stack_1:output:0=face_g_18/conv2d_transpose_100/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&face_g_18/conv2d_transpose_100/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@h
&face_g_18/conv2d_transpose_100/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@i
&face_g_18/conv2d_transpose_100/stack/3Const*
_output_shapes
: *
dtype0*
value
B :Ī
$face_g_18/conv2d_transpose_100/stackPack5face_g_18/conv2d_transpose_100/strided_slice:output:0/face_g_18/conv2d_transpose_100/stack/1:output:0/face_g_18/conv2d_transpose_100/stack/2:output:0/face_g_18/conv2d_transpose_100/stack/3:output:0*
N*
T0*
_output_shapes
:~
4face_g_18/conv2d_transpose_100/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6face_g_18/conv2d_transpose_100/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6face_g_18/conv2d_transpose_100/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
.face_g_18/conv2d_transpose_100/strided_slice_1StridedSlice-face_g_18/conv2d_transpose_100/stack:output:0=face_g_18/conv2d_transpose_100/strided_slice_1/stack:output:0?face_g_18/conv2d_transpose_100/strided_slice_1/stack_1:output:0?face_g_18/conv2d_transpose_100/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÐ
>face_g_18/conv2d_transpose_100/conv2d_transpose/ReadVariableOpReadVariableOpGface_g_18_conv2d_transpose_100_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Å
/face_g_18/conv2d_transpose_100/conv2d_transposeConv2DBackpropInput-face_g_18/conv2d_transpose_100/stack:output:0Fface_g_18/conv2d_transpose_100/conv2d_transpose/ReadVariableOp:value:0#face_g_18/LeakyRelu_6:activations:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
e
#face_g_18/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :û
face_g_18/concatenate_1/concatConcatV2#face_g_18/LeakyRelu_2:activations:08face_g_18/conv2d_transpose_100/conv2d_transpose:output:0,face_g_18/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ļ
*face_g_18/conv2d_614/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_614_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0å
face_g_18/conv2d_614/Conv2DConv2D'face_g_18/concatenate_1/concat:output:02face_g_18/conv2d_614/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

Iface_g_18/batch_instance_normalization_506/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ü
7face_g_18/batch_instance_normalization_506/moments/meanMean$face_g_18/conv2d_614/Conv2D:output:0Rface_g_18/batch_instance_normalization_506/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ã
?face_g_18/batch_instance_normalization_506/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_506/moments/mean:output:0*
T0*'
_output_shapes
:
Dface_g_18/batch_instance_normalization_506/moments/SquaredDifferenceSquaredDifference$face_g_18/conv2d_614/Conv2D:output:0Hface_g_18/batch_instance_normalization_506/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ē
Mface_g_18/batch_instance_normalization_506/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ļ
;face_g_18/batch_instance_normalization_506/moments/varianceMeanHface_g_18/batch_instance_normalization_506/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_506/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ø
.face_g_18/batch_instance_normalization_506/subSub$face_g_18/conv2d_614/Conv2D:output:0@face_g_18/batch_instance_normalization_506/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
0face_g_18/batch_instance_normalization_506/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ę
.face_g_18/batch_instance_normalization_506/addAddV2Dface_g_18/batch_instance_normalization_506/moments/variance:output:09face_g_18/batch_instance_normalization_506/add/y:output:0*
T0*'
_output_shapes
:
0face_g_18/batch_instance_normalization_506/RsqrtRsqrt2face_g_18/batch_instance_normalization_506/add:z:0*
T0*'
_output_shapes
:Ú
.face_g_18/batch_instance_normalization_506/mulMul2face_g_18/batch_instance_normalization_506/sub:z:04face_g_18/batch_instance_normalization_506/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Kface_g_18/batch_instance_normalization_506/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_506/moments_1/meanMean$face_g_18/conv2d_614/Conv2D:output:0Tface_g_18/batch_instance_normalization_506/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ð
Aface_g_18/batch_instance_normalization_506/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_506/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_506/moments_1/SquaredDifferenceSquaredDifference$face_g_18/conv2d_614/Conv2D:output:0Jface_g_18/batch_instance_normalization_506/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ 
Oface_g_18/batch_instance_normalization_506/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ·
=face_g_18/batch_instance_normalization_506/moments_1/varianceMeanJface_g_18/batch_instance_normalization_506/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_506/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ü
0face_g_18/batch_instance_normalization_506/sub_1Sub$face_g_18/conv2d_614/Conv2D:output:0Bface_g_18/batch_instance_normalization_506/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@w
2face_g_18/batch_instance_normalization_506/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ų
0face_g_18/batch_instance_normalization_506/add_1AddV2Fface_g_18/batch_instance_normalization_506/moments_1/variance:output:0;face_g_18/batch_instance_normalization_506/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸŽ
2face_g_18/batch_instance_normalization_506/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_506/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸā
0face_g_18/batch_instance_normalization_506/mul_1Mul4face_g_18/batch_instance_normalization_506/sub_1:z:06face_g_18/batch_instance_normalization_506/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@đ
9face_g_18/batch_instance_normalization_506/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_506_readvariableop_resource*
_output_shapes	
:*
dtype0é
0face_g_18/batch_instance_normalization_506/mul_2MulAface_g_18/batch_instance_normalization_506/ReadVariableOp:value:02face_g_18/batch_instance_normalization_506/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ŧ
;face_g_18/batch_instance_normalization_506/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_506_readvariableop_resource*
_output_shapes	
:*
dtype0w
2face_g_18/batch_instance_normalization_506/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ß
0face_g_18/batch_instance_normalization_506/sub_2Sub;face_g_18/batch_instance_normalization_506/sub_2/x:output:0Cface_g_18/batch_instance_normalization_506/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Þ
0face_g_18/batch_instance_normalization_506/mul_3Mul4face_g_18/batch_instance_normalization_506/sub_2:z:04face_g_18/batch_instance_normalization_506/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ā
0face_g_18/batch_instance_normalization_506/add_2AddV24face_g_18/batch_instance_normalization_506/mul_2:z:04face_g_18/batch_instance_normalization_506/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Å
?face_g_18/batch_instance_normalization_506/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_506_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0ņ
0face_g_18/batch_instance_normalization_506/mul_4Mul4face_g_18/batch_instance_normalization_506/add_2:z:0Gface_g_18/batch_instance_normalization_506/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Å
?face_g_18/batch_instance_normalization_506/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_506_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0ó
0face_g_18/batch_instance_normalization_506/add_3AddV24face_g_18/batch_instance_normalization_506/mul_4:z:0Gface_g_18/batch_instance_normalization_506/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
face_g_18/LeakyRelu_7	LeakyRelu4face_g_18/batch_instance_normalization_506/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ļ
*face_g_18/conv2d_615/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_615_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0á
face_g_18/conv2d_615/Conv2DConv2D#face_g_18/LeakyRelu_7:activations:02face_g_18/conv2d_615/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

Iface_g_18/batch_instance_normalization_507/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          ü
7face_g_18/batch_instance_normalization_507/moments/meanMean$face_g_18/conv2d_615/Conv2D:output:0Rface_g_18/batch_instance_normalization_507/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ã
?face_g_18/batch_instance_normalization_507/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_507/moments/mean:output:0*
T0*'
_output_shapes
:
Dface_g_18/batch_instance_normalization_507/moments/SquaredDifferenceSquaredDifference$face_g_18/conv2d_615/Conv2D:output:0Hface_g_18/batch_instance_normalization_507/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ē
Mface_g_18/batch_instance_normalization_507/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ļ
;face_g_18/batch_instance_normalization_507/moments/varianceMeanHface_g_18/batch_instance_normalization_507/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_507/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ø
.face_g_18/batch_instance_normalization_507/subSub$face_g_18/conv2d_615/Conv2D:output:0@face_g_18/batch_instance_normalization_507/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
0face_g_18/batch_instance_normalization_507/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ę
.face_g_18/batch_instance_normalization_507/addAddV2Dface_g_18/batch_instance_normalization_507/moments/variance:output:09face_g_18/batch_instance_normalization_507/add/y:output:0*
T0*'
_output_shapes
:
0face_g_18/batch_instance_normalization_507/RsqrtRsqrt2face_g_18/batch_instance_normalization_507/add:z:0*
T0*'
_output_shapes
:Ú
.face_g_18/batch_instance_normalization_507/mulMul2face_g_18/batch_instance_normalization_507/sub:z:04face_g_18/batch_instance_normalization_507/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Kface_g_18/batch_instance_normalization_507/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_507/moments_1/meanMean$face_g_18/conv2d_615/Conv2D:output:0Tface_g_18/batch_instance_normalization_507/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ð
Aface_g_18/batch_instance_normalization_507/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_507/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_507/moments_1/SquaredDifferenceSquaredDifference$face_g_18/conv2d_615/Conv2D:output:0Jface_g_18/batch_instance_normalization_507/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ 
Oface_g_18/batch_instance_normalization_507/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ·
=face_g_18/batch_instance_normalization_507/moments_1/varianceMeanJface_g_18/batch_instance_normalization_507/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_507/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ü
0face_g_18/batch_instance_normalization_507/sub_1Sub$face_g_18/conv2d_615/Conv2D:output:0Bface_g_18/batch_instance_normalization_507/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@w
2face_g_18/batch_instance_normalization_507/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ų
0face_g_18/batch_instance_normalization_507/add_1AddV2Fface_g_18/batch_instance_normalization_507/moments_1/variance:output:0;face_g_18/batch_instance_normalization_507/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸŽ
2face_g_18/batch_instance_normalization_507/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_507/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸā
0face_g_18/batch_instance_normalization_507/mul_1Mul4face_g_18/batch_instance_normalization_507/sub_1:z:06face_g_18/batch_instance_normalization_507/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@đ
9face_g_18/batch_instance_normalization_507/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_507_readvariableop_resource*
_output_shapes	
:*
dtype0é
0face_g_18/batch_instance_normalization_507/mul_2MulAface_g_18/batch_instance_normalization_507/ReadVariableOp:value:02face_g_18/batch_instance_normalization_507/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ŧ
;face_g_18/batch_instance_normalization_507/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_507_readvariableop_resource*
_output_shapes	
:*
dtype0w
2face_g_18/batch_instance_normalization_507/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ß
0face_g_18/batch_instance_normalization_507/sub_2Sub;face_g_18/batch_instance_normalization_507/sub_2/x:output:0Cface_g_18/batch_instance_normalization_507/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Þ
0face_g_18/batch_instance_normalization_507/mul_3Mul4face_g_18/batch_instance_normalization_507/sub_2:z:04face_g_18/batch_instance_normalization_507/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ā
0face_g_18/batch_instance_normalization_507/add_2AddV24face_g_18/batch_instance_normalization_507/mul_2:z:04face_g_18/batch_instance_normalization_507/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Å
?face_g_18/batch_instance_normalization_507/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_507_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0ņ
0face_g_18/batch_instance_normalization_507/mul_4Mul4face_g_18/batch_instance_normalization_507/add_2:z:0Gface_g_18/batch_instance_normalization_507/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Å
?face_g_18/batch_instance_normalization_507/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_507_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0ó
0face_g_18/batch_instance_normalization_507/add_3AddV24face_g_18/batch_instance_normalization_507/mul_4:z:0Gface_g_18/batch_instance_normalization_507/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
face_g_18/LeakyRelu_8	LeakyRelu4face_g_18/batch_instance_normalization_507/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@w
$face_g_18/conv2d_transpose_101/ShapeShape#face_g_18/LeakyRelu_8:activations:0*
T0*
_output_shapes
:|
2face_g_18/conv2d_transpose_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4face_g_18/conv2d_transpose_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4face_g_18/conv2d_transpose_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ė
,face_g_18/conv2d_transpose_101/strided_sliceStridedSlice-face_g_18/conv2d_transpose_101/Shape:output:0;face_g_18/conv2d_transpose_101/strided_slice/stack:output:0=face_g_18/conv2d_transpose_101/strided_slice/stack_1:output:0=face_g_18/conv2d_transpose_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
&face_g_18/conv2d_transpose_101/stack/1Const*
_output_shapes
: *
dtype0*
value
B :i
&face_g_18/conv2d_transpose_101/stack/2Const*
_output_shapes
: *
dtype0*
value
B :h
&face_g_18/conv2d_transpose_101/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@Ī
$face_g_18/conv2d_transpose_101/stackPack5face_g_18/conv2d_transpose_101/strided_slice:output:0/face_g_18/conv2d_transpose_101/stack/1:output:0/face_g_18/conv2d_transpose_101/stack/2:output:0/face_g_18/conv2d_transpose_101/stack/3:output:0*
N*
T0*
_output_shapes
:~
4face_g_18/conv2d_transpose_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
6face_g_18/conv2d_transpose_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
6face_g_18/conv2d_transpose_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
.face_g_18/conv2d_transpose_101/strided_slice_1StridedSlice-face_g_18/conv2d_transpose_101/stack:output:0=face_g_18/conv2d_transpose_101/strided_slice_1/stack:output:0?face_g_18/conv2d_transpose_101/strided_slice_1/stack_1:output:0?face_g_18/conv2d_transpose_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÏ
>face_g_18/conv2d_transpose_101/conv2d_transpose/ReadVariableOpReadVariableOpGface_g_18_conv2d_transpose_101_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0Æ
/face_g_18/conv2d_transpose_101/conv2d_transposeConv2DBackpropInput-face_g_18/conv2d_transpose_101/stack:output:0Fface_g_18/conv2d_transpose_101/conv2d_transpose/ReadVariableOp:value:0#face_g_18/LeakyRelu_8:activations:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
e
#face_g_18/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :û
face_g_18/concatenate_2/concatConcatV2!face_g_18/LeakyRelu:activations:08face_g_18/conv2d_transpose_101/conv2d_transpose:output:0,face_g_18/concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ§
*face_g_18/conv2d_616/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_616_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0æ
face_g_18/conv2d_616/Conv2DConv2D'face_g_18/concatenate_2/concat:output:02face_g_18/conv2d_616/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

Iface_g_18/batch_instance_normalization_508/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          û
7face_g_18/batch_instance_normalization_508/moments/meanMean$face_g_18/conv2d_616/Conv2D:output:0Rface_g_18/batch_instance_normalization_508/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(Â
?face_g_18/batch_instance_normalization_508/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_508/moments/mean:output:0*
T0*&
_output_shapes
:@
Dface_g_18/batch_instance_normalization_508/moments/SquaredDifferenceSquaredDifference$face_g_18/conv2d_616/Conv2D:output:0Hface_g_18/batch_instance_normalization_508/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ē
Mface_g_18/batch_instance_normalization_508/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
;face_g_18/batch_instance_normalization_508/moments/varianceMeanHface_g_18/batch_instance_normalization_508/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_508/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(Ų
.face_g_18/batch_instance_normalization_508/subSub$face_g_18/conv2d_616/Conv2D:output:0@face_g_18/batch_instance_normalization_508/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@u
0face_g_18/batch_instance_normalization_508/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7é
.face_g_18/batch_instance_normalization_508/addAddV2Dface_g_18/batch_instance_normalization_508/moments/variance:output:09face_g_18/batch_instance_normalization_508/add/y:output:0*
T0*&
_output_shapes
:@
0face_g_18/batch_instance_normalization_508/RsqrtRsqrt2face_g_18/batch_instance_normalization_508/add:z:0*
T0*&
_output_shapes
:@Û
.face_g_18/batch_instance_normalization_508/mulMul2face_g_18/batch_instance_normalization_508/sub:z:04face_g_18/batch_instance_normalization_508/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Kface_g_18/batch_instance_normalization_508/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_508/moments_1/meanMean$face_g_18/conv2d_616/Conv2D:output:0Tface_g_18/batch_instance_normalization_508/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(Ï
Aface_g_18/batch_instance_normalization_508/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_508/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Fface_g_18/batch_instance_normalization_508/moments_1/SquaredDifferenceSquaredDifference$face_g_18/conv2d_616/Conv2D:output:0Jface_g_18/batch_instance_normalization_508/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ 
Oface_g_18/batch_instance_normalization_508/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
=face_g_18/batch_instance_normalization_508/moments_1/varianceMeanJface_g_18/batch_instance_normalization_508/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_508/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(Ý
0face_g_18/batch_instance_normalization_508/sub_1Sub$face_g_18/conv2d_616/Conv2D:output:0Bface_g_18/batch_instance_normalization_508/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@w
2face_g_18/batch_instance_normalization_508/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ø
0face_g_18/batch_instance_normalization_508/add_1AddV2Fface_g_18/batch_instance_normalization_508/moments_1/variance:output:0;face_g_18/batch_instance_normalization_508/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ŧ
2face_g_18/batch_instance_normalization_508/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_508/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@á
0face_g_18/batch_instance_normalization_508/mul_1Mul4face_g_18/batch_instance_normalization_508/sub_1:z:06face_g_18/batch_instance_normalization_508/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ļ
9face_g_18/batch_instance_normalization_508/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_508_readvariableop_resource*
_output_shapes
:@*
dtype0ę
0face_g_18/batch_instance_normalization_508/mul_2MulAface_g_18/batch_instance_normalization_508/ReadVariableOp:value:02face_g_18/batch_instance_normalization_508/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@š
;face_g_18/batch_instance_normalization_508/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_508_readvariableop_resource*
_output_shapes
:@*
dtype0w
2face_g_18/batch_instance_normalization_508/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Þ
0face_g_18/batch_instance_normalization_508/sub_2Sub;face_g_18/batch_instance_normalization_508/sub_2/x:output:0Cface_g_18/batch_instance_normalization_508/ReadVariableOp_1:value:0*
T0*
_output_shapes
:@ß
0face_g_18/batch_instance_normalization_508/mul_3Mul4face_g_18/batch_instance_normalization_508/sub_2:z:04face_g_18/batch_instance_normalization_508/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@á
0face_g_18/batch_instance_normalization_508/add_2AddV24face_g_18/batch_instance_normalization_508/mul_2:z:04face_g_18/batch_instance_normalization_508/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ä
?face_g_18/batch_instance_normalization_508/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_508_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0ō
0face_g_18/batch_instance_normalization_508/mul_4Mul4face_g_18/batch_instance_normalization_508/add_2:z:0Gface_g_18/batch_instance_normalization_508/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ä
?face_g_18/batch_instance_normalization_508/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_508_add_3_readvariableop_resource*
_output_shapes
:@*
dtype0ô
0face_g_18/batch_instance_normalization_508/add_3AddV24face_g_18/batch_instance_normalization_508/mul_4:z:0Gface_g_18/batch_instance_normalization_508/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
face_g_18/LeakyRelu_9	LeakyRelu4face_g_18/batch_instance_normalization_508/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ķ
*face_g_18/conv2d_617/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_617_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0â
face_g_18/conv2d_617/Conv2DConv2D#face_g_18/LeakyRelu_9:activations:02face_g_18/conv2d_617/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

Iface_g_18/batch_instance_normalization_509/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          û
7face_g_18/batch_instance_normalization_509/moments/meanMean$face_g_18/conv2d_617/Conv2D:output:0Rface_g_18/batch_instance_normalization_509/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(Â
?face_g_18/batch_instance_normalization_509/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_509/moments/mean:output:0*
T0*&
_output_shapes
:@
Dface_g_18/batch_instance_normalization_509/moments/SquaredDifferenceSquaredDifference$face_g_18/conv2d_617/Conv2D:output:0Hface_g_18/batch_instance_normalization_509/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ē
Mface_g_18/batch_instance_normalization_509/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
;face_g_18/batch_instance_normalization_509/moments/varianceMeanHface_g_18/batch_instance_normalization_509/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_509/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(Ų
.face_g_18/batch_instance_normalization_509/subSub$face_g_18/conv2d_617/Conv2D:output:0@face_g_18/batch_instance_normalization_509/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@u
0face_g_18/batch_instance_normalization_509/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7é
.face_g_18/batch_instance_normalization_509/addAddV2Dface_g_18/batch_instance_normalization_509/moments/variance:output:09face_g_18/batch_instance_normalization_509/add/y:output:0*
T0*&
_output_shapes
:@
0face_g_18/batch_instance_normalization_509/RsqrtRsqrt2face_g_18/batch_instance_normalization_509/add:z:0*
T0*&
_output_shapes
:@Û
.face_g_18/batch_instance_normalization_509/mulMul2face_g_18/batch_instance_normalization_509/sub:z:04face_g_18/batch_instance_normalization_509/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Kface_g_18/batch_instance_normalization_509/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_509/moments_1/meanMean$face_g_18/conv2d_617/Conv2D:output:0Tface_g_18/batch_instance_normalization_509/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(Ï
Aface_g_18/batch_instance_normalization_509/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_509/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Fface_g_18/batch_instance_normalization_509/moments_1/SquaredDifferenceSquaredDifference$face_g_18/conv2d_617/Conv2D:output:0Jface_g_18/batch_instance_normalization_509/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ 
Oface_g_18/batch_instance_normalization_509/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
=face_g_18/batch_instance_normalization_509/moments_1/varianceMeanJface_g_18/batch_instance_normalization_509/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_509/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(Ý
0face_g_18/batch_instance_normalization_509/sub_1Sub$face_g_18/conv2d_617/Conv2D:output:0Bface_g_18/batch_instance_normalization_509/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@w
2face_g_18/batch_instance_normalization_509/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ø
0face_g_18/batch_instance_normalization_509/add_1AddV2Fface_g_18/batch_instance_normalization_509/moments_1/variance:output:0;face_g_18/batch_instance_normalization_509/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ŧ
2face_g_18/batch_instance_normalization_509/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_509/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@á
0face_g_18/batch_instance_normalization_509/mul_1Mul4face_g_18/batch_instance_normalization_509/sub_1:z:06face_g_18/batch_instance_normalization_509/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ļ
9face_g_18/batch_instance_normalization_509/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_509_readvariableop_resource*
_output_shapes
:@*
dtype0ę
0face_g_18/batch_instance_normalization_509/mul_2MulAface_g_18/batch_instance_normalization_509/ReadVariableOp:value:02face_g_18/batch_instance_normalization_509/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@š
;face_g_18/batch_instance_normalization_509/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_509_readvariableop_resource*
_output_shapes
:@*
dtype0w
2face_g_18/batch_instance_normalization_509/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Þ
0face_g_18/batch_instance_normalization_509/sub_2Sub;face_g_18/batch_instance_normalization_509/sub_2/x:output:0Cface_g_18/batch_instance_normalization_509/ReadVariableOp_1:value:0*
T0*
_output_shapes
:@ß
0face_g_18/batch_instance_normalization_509/mul_3Mul4face_g_18/batch_instance_normalization_509/sub_2:z:04face_g_18/batch_instance_normalization_509/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@á
0face_g_18/batch_instance_normalization_509/add_2AddV24face_g_18/batch_instance_normalization_509/mul_2:z:04face_g_18/batch_instance_normalization_509/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ä
?face_g_18/batch_instance_normalization_509/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_509_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0ō
0face_g_18/batch_instance_normalization_509/mul_4Mul4face_g_18/batch_instance_normalization_509/add_2:z:0Gface_g_18/batch_instance_normalization_509/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ä
?face_g_18/batch_instance_normalization_509/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_509_add_3_readvariableop_resource*
_output_shapes
:@*
dtype0ô
0face_g_18/batch_instance_normalization_509/add_3AddV24face_g_18/batch_instance_normalization_509/mul_4:z:0Gface_g_18/batch_instance_normalization_509/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
face_g_18/LeakyRelu_10	LeakyRelu4face_g_18/batch_instance_normalization_509/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ķ
*face_g_18/conv2d_618/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_618_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0ã
face_g_18/conv2d_618/Conv2DConv2D$face_g_18/LeakyRelu_10:activations:02face_g_18/conv2d_618/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides

Iface_g_18/batch_instance_normalization_510/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          û
7face_g_18/batch_instance_normalization_510/moments/meanMean$face_g_18/conv2d_618/Conv2D:output:0Rface_g_18/batch_instance_normalization_510/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(Â
?face_g_18/batch_instance_normalization_510/moments/StopGradientStopGradient@face_g_18/batch_instance_normalization_510/moments/mean:output:0*
T0*&
_output_shapes
:
Dface_g_18/batch_instance_normalization_510/moments/SquaredDifferenceSquaredDifference$face_g_18/conv2d_618/Conv2D:output:0Hface_g_18/batch_instance_normalization_510/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸĒ
Mface_g_18/batch_instance_normalization_510/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          §
;face_g_18/batch_instance_normalization_510/moments/varianceMeanHface_g_18/batch_instance_normalization_510/moments/SquaredDifference:z:0Vface_g_18/batch_instance_normalization_510/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(Ų
.face_g_18/batch_instance_normalization_510/subSub$face_g_18/conv2d_618/Conv2D:output:0@face_g_18/batch_instance_normalization_510/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸu
0face_g_18/batch_instance_normalization_510/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7é
.face_g_18/batch_instance_normalization_510/addAddV2Dface_g_18/batch_instance_normalization_510/moments/variance:output:09face_g_18/batch_instance_normalization_510/add/y:output:0*
T0*&
_output_shapes
:
0face_g_18/batch_instance_normalization_510/RsqrtRsqrt2face_g_18/batch_instance_normalization_510/add:z:0*
T0*&
_output_shapes
:Û
.face_g_18/batch_instance_normalization_510/mulMul2face_g_18/batch_instance_normalization_510/sub:z:04face_g_18/batch_instance_normalization_510/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Kface_g_18/batch_instance_normalization_510/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
9face_g_18/batch_instance_normalization_510/moments_1/meanMean$face_g_18/conv2d_618/Conv2D:output:0Tface_g_18/batch_instance_normalization_510/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ï
Aface_g_18/batch_instance_normalization_510/moments_1/StopGradientStopGradientBface_g_18/batch_instance_normalization_510/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Fface_g_18/batch_instance_normalization_510/moments_1/SquaredDifferenceSquaredDifference$face_g_18/conv2d_618/Conv2D:output:0Jface_g_18/batch_instance_normalization_510/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ 
Oface_g_18/batch_instance_normalization_510/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
=face_g_18/batch_instance_normalization_510/moments_1/varianceMeanJface_g_18/batch_instance_normalization_510/moments_1/SquaredDifference:z:0Xface_g_18/batch_instance_normalization_510/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Ý
0face_g_18/batch_instance_normalization_510/sub_1Sub$face_g_18/conv2d_618/Conv2D:output:0Bface_g_18/batch_instance_normalization_510/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸw
2face_g_18/batch_instance_normalization_510/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7ø
0face_g_18/batch_instance_normalization_510/add_1AddV2Fface_g_18/batch_instance_normalization_510/moments_1/variance:output:0;face_g_18/batch_instance_normalization_510/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸŦ
2face_g_18/batch_instance_normalization_510/Rsqrt_1Rsqrt4face_g_18/batch_instance_normalization_510/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸá
0face_g_18/batch_instance_normalization_510/mul_1Mul4face_g_18/batch_instance_normalization_510/sub_1:z:06face_g_18/batch_instance_normalization_510/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸļ
9face_g_18/batch_instance_normalization_510/ReadVariableOpReadVariableOpBface_g_18_batch_instance_normalization_510_readvariableop_resource*
_output_shapes
:*
dtype0ę
0face_g_18/batch_instance_normalization_510/mul_2MulAface_g_18/batch_instance_normalization_510/ReadVariableOp:value:02face_g_18/batch_instance_normalization_510/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸš
;face_g_18/batch_instance_normalization_510/ReadVariableOp_1ReadVariableOpBface_g_18_batch_instance_normalization_510_readvariableop_resource*
_output_shapes
:*
dtype0w
2face_g_18/batch_instance_normalization_510/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Þ
0face_g_18/batch_instance_normalization_510/sub_2Sub;face_g_18/batch_instance_normalization_510/sub_2/x:output:0Cface_g_18/batch_instance_normalization_510/ReadVariableOp_1:value:0*
T0*
_output_shapes
:ß
0face_g_18/batch_instance_normalization_510/mul_3Mul4face_g_18/batch_instance_normalization_510/sub_2:z:04face_g_18/batch_instance_normalization_510/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸá
0face_g_18/batch_instance_normalization_510/add_2AddV24face_g_18/batch_instance_normalization_510/mul_2:z:04face_g_18/batch_instance_normalization_510/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸÄ
?face_g_18/batch_instance_normalization_510/mul_4/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_510_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0ō
0face_g_18/batch_instance_normalization_510/mul_4Mul4face_g_18/batch_instance_normalization_510/add_2:z:0Gface_g_18/batch_instance_normalization_510/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸÄ
?face_g_18/batch_instance_normalization_510/add_3/ReadVariableOpReadVariableOpHface_g_18_batch_instance_normalization_510_add_3_readvariableop_resource*
_output_shapes
:*
dtype0ô
0face_g_18/batch_instance_normalization_510/add_3AddV24face_g_18/batch_instance_normalization_510/mul_4:z:0Gface_g_18/batch_instance_normalization_510/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
face_g_18/LeakyRelu_11	LeakyRelu4face_g_18/batch_instance_normalization_510/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸĶ
*face_g_18/conv2d_619/Conv2D/ReadVariableOpReadVariableOp3face_g_18_conv2d_619_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0ã
face_g_18/conv2d_619/Conv2DConv2D$face_g_18/LeakyRelu_11:activations:02face_g_18/conv2d_619/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides
x
face_g_18/TanhTanh$face_g_18/conv2d_619/Conv2D:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸk
IdentityIdentityface_g_18/Tanh:y:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸĻ
NoOpNoOp:^face_g_18/batch_instance_normalization_500/ReadVariableOp<^face_g_18/batch_instance_normalization_500/ReadVariableOp_1@^face_g_18/batch_instance_normalization_500/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_500/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_501/ReadVariableOp<^face_g_18/batch_instance_normalization_501/ReadVariableOp_1@^face_g_18/batch_instance_normalization_501/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_501/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_502/ReadVariableOp<^face_g_18/batch_instance_normalization_502/ReadVariableOp_1@^face_g_18/batch_instance_normalization_502/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_502/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_503/ReadVariableOp<^face_g_18/batch_instance_normalization_503/ReadVariableOp_1@^face_g_18/batch_instance_normalization_503/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_503/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_504/ReadVariableOp<^face_g_18/batch_instance_normalization_504/ReadVariableOp_1@^face_g_18/batch_instance_normalization_504/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_504/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_505/ReadVariableOp<^face_g_18/batch_instance_normalization_505/ReadVariableOp_1@^face_g_18/batch_instance_normalization_505/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_505/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_506/ReadVariableOp<^face_g_18/batch_instance_normalization_506/ReadVariableOp_1@^face_g_18/batch_instance_normalization_506/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_506/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_507/ReadVariableOp<^face_g_18/batch_instance_normalization_507/ReadVariableOp_1@^face_g_18/batch_instance_normalization_507/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_507/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_508/ReadVariableOp<^face_g_18/batch_instance_normalization_508/ReadVariableOp_1@^face_g_18/batch_instance_normalization_508/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_508/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_509/ReadVariableOp<^face_g_18/batch_instance_normalization_509/ReadVariableOp_1@^face_g_18/batch_instance_normalization_509/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_509/mul_4/ReadVariableOp:^face_g_18/batch_instance_normalization_510/ReadVariableOp<^face_g_18/batch_instance_normalization_510/ReadVariableOp_1@^face_g_18/batch_instance_normalization_510/add_3/ReadVariableOp@^face_g_18/batch_instance_normalization_510/mul_4/ReadVariableOp+^face_g_18/conv2d_606/Conv2D/ReadVariableOp+^face_g_18/conv2d_607/Conv2D/ReadVariableOp+^face_g_18/conv2d_608/Conv2D/ReadVariableOp+^face_g_18/conv2d_609/Conv2D/ReadVariableOp+^face_g_18/conv2d_610/Conv2D/ReadVariableOp+^face_g_18/conv2d_611/Conv2D/ReadVariableOp+^face_g_18/conv2d_612/Conv2D/ReadVariableOp+^face_g_18/conv2d_613/Conv2D/ReadVariableOp+^face_g_18/conv2d_614/Conv2D/ReadVariableOp+^face_g_18/conv2d_615/Conv2D/ReadVariableOp+^face_g_18/conv2d_616/Conv2D/ReadVariableOp+^face_g_18/conv2d_617/Conv2D/ReadVariableOp+^face_g_18/conv2d_618/Conv2D/ReadVariableOp+^face_g_18/conv2d_619/Conv2D/ReadVariableOp?^face_g_18/conv2d_transpose_100/conv2d_transpose/ReadVariableOp?^face_g_18/conv2d_transpose_101/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2v
9face_g_18/batch_instance_normalization_500/ReadVariableOp9face_g_18/batch_instance_normalization_500/ReadVariableOp2z
;face_g_18/batch_instance_normalization_500/ReadVariableOp_1;face_g_18/batch_instance_normalization_500/ReadVariableOp_12
?face_g_18/batch_instance_normalization_500/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_500/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_500/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_500/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_501/ReadVariableOp9face_g_18/batch_instance_normalization_501/ReadVariableOp2z
;face_g_18/batch_instance_normalization_501/ReadVariableOp_1;face_g_18/batch_instance_normalization_501/ReadVariableOp_12
?face_g_18/batch_instance_normalization_501/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_501/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_501/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_501/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_502/ReadVariableOp9face_g_18/batch_instance_normalization_502/ReadVariableOp2z
;face_g_18/batch_instance_normalization_502/ReadVariableOp_1;face_g_18/batch_instance_normalization_502/ReadVariableOp_12
?face_g_18/batch_instance_normalization_502/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_502/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_502/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_502/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_503/ReadVariableOp9face_g_18/batch_instance_normalization_503/ReadVariableOp2z
;face_g_18/batch_instance_normalization_503/ReadVariableOp_1;face_g_18/batch_instance_normalization_503/ReadVariableOp_12
?face_g_18/batch_instance_normalization_503/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_503/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_503/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_503/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_504/ReadVariableOp9face_g_18/batch_instance_normalization_504/ReadVariableOp2z
;face_g_18/batch_instance_normalization_504/ReadVariableOp_1;face_g_18/batch_instance_normalization_504/ReadVariableOp_12
?face_g_18/batch_instance_normalization_504/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_504/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_504/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_504/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_505/ReadVariableOp9face_g_18/batch_instance_normalization_505/ReadVariableOp2z
;face_g_18/batch_instance_normalization_505/ReadVariableOp_1;face_g_18/batch_instance_normalization_505/ReadVariableOp_12
?face_g_18/batch_instance_normalization_505/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_505/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_505/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_505/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_506/ReadVariableOp9face_g_18/batch_instance_normalization_506/ReadVariableOp2z
;face_g_18/batch_instance_normalization_506/ReadVariableOp_1;face_g_18/batch_instance_normalization_506/ReadVariableOp_12
?face_g_18/batch_instance_normalization_506/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_506/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_506/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_506/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_507/ReadVariableOp9face_g_18/batch_instance_normalization_507/ReadVariableOp2z
;face_g_18/batch_instance_normalization_507/ReadVariableOp_1;face_g_18/batch_instance_normalization_507/ReadVariableOp_12
?face_g_18/batch_instance_normalization_507/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_507/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_507/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_507/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_508/ReadVariableOp9face_g_18/batch_instance_normalization_508/ReadVariableOp2z
;face_g_18/batch_instance_normalization_508/ReadVariableOp_1;face_g_18/batch_instance_normalization_508/ReadVariableOp_12
?face_g_18/batch_instance_normalization_508/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_508/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_508/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_508/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_509/ReadVariableOp9face_g_18/batch_instance_normalization_509/ReadVariableOp2z
;face_g_18/batch_instance_normalization_509/ReadVariableOp_1;face_g_18/batch_instance_normalization_509/ReadVariableOp_12
?face_g_18/batch_instance_normalization_509/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_509/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_509/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_509/mul_4/ReadVariableOp2v
9face_g_18/batch_instance_normalization_510/ReadVariableOp9face_g_18/batch_instance_normalization_510/ReadVariableOp2z
;face_g_18/batch_instance_normalization_510/ReadVariableOp_1;face_g_18/batch_instance_normalization_510/ReadVariableOp_12
?face_g_18/batch_instance_normalization_510/add_3/ReadVariableOp?face_g_18/batch_instance_normalization_510/add_3/ReadVariableOp2
?face_g_18/batch_instance_normalization_510/mul_4/ReadVariableOp?face_g_18/batch_instance_normalization_510/mul_4/ReadVariableOp2X
*face_g_18/conv2d_606/Conv2D/ReadVariableOp*face_g_18/conv2d_606/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_607/Conv2D/ReadVariableOp*face_g_18/conv2d_607/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_608/Conv2D/ReadVariableOp*face_g_18/conv2d_608/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_609/Conv2D/ReadVariableOp*face_g_18/conv2d_609/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_610/Conv2D/ReadVariableOp*face_g_18/conv2d_610/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_611/Conv2D/ReadVariableOp*face_g_18/conv2d_611/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_612/Conv2D/ReadVariableOp*face_g_18/conv2d_612/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_613/Conv2D/ReadVariableOp*face_g_18/conv2d_613/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_614/Conv2D/ReadVariableOp*face_g_18/conv2d_614/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_615/Conv2D/ReadVariableOp*face_g_18/conv2d_615/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_616/Conv2D/ReadVariableOp*face_g_18/conv2d_616/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_617/Conv2D/ReadVariableOp*face_g_18/conv2d_617/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_618/Conv2D/ReadVariableOp*face_g_18/conv2d_618/Conv2D/ReadVariableOp2X
*face_g_18/conv2d_619/Conv2D/ReadVariableOp*face_g_18/conv2d_619/Conv2D/ReadVariableOp2
>face_g_18/conv2d_transpose_100/conv2d_transpose/ReadVariableOp>face_g_18/conv2d_transpose_100/conv2d_transpose/ReadVariableOp2
>face_g_18/conv2d_transpose_101/conv2d_transpose/ReadVariableOp>face_g_18/conv2d_transpose_101/conv2d_transpose/ReadVariableOp:Z V
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_1:ZV
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
!
_user_specified_name	input_2
ģ
ŧ
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56542114

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOpe
Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      l
Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            m
Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
3Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
0Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
-Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                r
!Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ŋ
Conv2D/SpaceToBatchNDSpaceToBatchNDinputs*Conv2D/SpaceToBatchND/block_shape:output:0'Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ģ
Conv2DConv2DConv2D/SpaceToBatchND:output:0Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
r
!Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      |
Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                Å
Conv2D/BatchToSpaceNDBatchToSpaceNDConv2D:output:0*Conv2D/BatchToSpaceND/block_shape:output:0$Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
IdentityIdentityConv2D/BatchToSpaceND:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ  : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
 
_user_specified_nameinputs

Í
,__inference_face_g_18_layer_call_fn_56543776
inputs_0
inputs_1!
unknown:@#
	unknown_0:@@$
	unknown_1:@
	unknown_2:	
	unknown_3:	
	unknown_4:	%
	unknown_5:
	unknown_6:	
	unknown_7:	
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	&

unknown_13:

unknown_14:	

unknown_15:	

unknown_16:	&

unknown_17:

unknown_18:	

unknown_19:	

unknown_20:	&

unknown_21:

unknown_22:	

unknown_23:	

unknown_24:	&

unknown_25:&

unknown_26:

unknown_27:	

unknown_28:	

unknown_29:	&

unknown_30:

unknown_31:	

unknown_32:	

unknown_33:	%

unknown_34:@%

unknown_35:@

unknown_36:@

unknown_37:@

unknown_38:@$

unknown_39:@@

unknown_40:@

unknown_41:@

unknown_42:@$

unknown_43:@

unknown_44:

unknown_45:

unknown_46:$

unknown_47:
identityĒStatefulPartitionedCallĸ
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47*>
Tin7
523*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*S
_read_only_resource_inputs5
31	
 !"#$%&'()*+,-./012*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_face_g_18_layer_call_and_return_conditional_losses_56542562y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"
_user_specified_name
inputs/1

k
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56545112

inputs
identityĒ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ:r n
J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Š
ŧ
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56545611

inputs:
conv2d_readvariableop_resource:
identityĒConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
g
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:ĸĸĸĸĸĸĸĸĸ@@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56545560
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ  w
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
:ĸĸĸĸĸĸĸĸĸ  J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ  q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  e
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
:ĸĸĸĸĸĸĸĸĸ  _
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ  : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  

_user_specified_namex
Į
Ų
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56545597

inputsD
(conv2d_transpose_readvariableop_resource:
identityĒconv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
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
valueB:Ņ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ų
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0Ý
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides

IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸh
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
°

7__inference_conv2d_transpose_100_layer_call_fn_56545567

inputs#
unknown:
identityĒStatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56541751
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ: 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Õ

-__inference_conv2d_617_layer_call_fn_56545836

inputs!
unknown:@@
identityĒStatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56542436y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
·$
Î
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56541881
x&
readvariableop_resource:	,
mul_4_readvariableop_resource:	,
add_3_readvariableop_resource:	
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
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
:ĸĸĸĸĸĸĸĸĸ@@w
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
:ĸĸĸĸĸĸĸĸĸ@@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7i
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
:ĸĸĸĸĸĸĸĸĸ@@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(z
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ķ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(c
sub_1Subxmoments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7x
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸV
Rsqrt_1Rsqrt	add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ_
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0h
mul_2MulReadVariableOp:value:0mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@e
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
:ĸĸĸĸĸĸĸĸĸ@@_
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes	
:*
dtype0p
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@o
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes	
:*
dtype0r
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@a
IdentityIdentity	add_3:z:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex
ķ
Ā
C__inference_batch_instance_normalization_510_layer_call_fn_56545919
x
unknown:
	unknown_0:
	unknown_1:
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56542540y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ

_user_specified_namex
ëģ
ï4
G__inference_face_g_18_layer_call_and_return_conditional_losses_56544424
inputs_0
inputs_1C
)conv2d_606_conv2d_readvariableop_resource:@C
)conv2d_607_conv2d_readvariableop_resource:@@D
)conv2d_608_conv2d_readvariableop_resource:@G
8batch_instance_normalization_500_readvariableop_resource:	M
>batch_instance_normalization_500_mul_4_readvariableop_resource:	M
>batch_instance_normalization_500_add_3_readvariableop_resource:	E
)conv2d_609_conv2d_readvariableop_resource:G
8batch_instance_normalization_501_readvariableop_resource:	M
>batch_instance_normalization_501_mul_4_readvariableop_resource:	M
>batch_instance_normalization_501_add_3_readvariableop_resource:	E
)conv2d_610_conv2d_readvariableop_resource:G
8batch_instance_normalization_502_readvariableop_resource:	M
>batch_instance_normalization_502_mul_4_readvariableop_resource:	M
>batch_instance_normalization_502_add_3_readvariableop_resource:	E
)conv2d_611_conv2d_readvariableop_resource:G
8batch_instance_normalization_503_readvariableop_resource:	M
>batch_instance_normalization_503_mul_4_readvariableop_resource:	M
>batch_instance_normalization_503_add_3_readvariableop_resource:	E
)conv2d_612_conv2d_readvariableop_resource:G
8batch_instance_normalization_504_readvariableop_resource:	M
>batch_instance_normalization_504_mul_4_readvariableop_resource:	M
>batch_instance_normalization_504_add_3_readvariableop_resource:	E
)conv2d_613_conv2d_readvariableop_resource:G
8batch_instance_normalization_505_readvariableop_resource:	M
>batch_instance_normalization_505_mul_4_readvariableop_resource:	M
>batch_instance_normalization_505_add_3_readvariableop_resource:	Y
=conv2d_transpose_100_conv2d_transpose_readvariableop_resource:E
)conv2d_614_conv2d_readvariableop_resource:G
8batch_instance_normalization_506_readvariableop_resource:	M
>batch_instance_normalization_506_mul_4_readvariableop_resource:	M
>batch_instance_normalization_506_add_3_readvariableop_resource:	E
)conv2d_615_conv2d_readvariableop_resource:G
8batch_instance_normalization_507_readvariableop_resource:	M
>batch_instance_normalization_507_mul_4_readvariableop_resource:	M
>batch_instance_normalization_507_add_3_readvariableop_resource:	X
=conv2d_transpose_101_conv2d_transpose_readvariableop_resource:@D
)conv2d_616_conv2d_readvariableop_resource:@F
8batch_instance_normalization_508_readvariableop_resource:@L
>batch_instance_normalization_508_mul_4_readvariableop_resource:@L
>batch_instance_normalization_508_add_3_readvariableop_resource:@C
)conv2d_617_conv2d_readvariableop_resource:@@F
8batch_instance_normalization_509_readvariableop_resource:@L
>batch_instance_normalization_509_mul_4_readvariableop_resource:@L
>batch_instance_normalization_509_add_3_readvariableop_resource:@C
)conv2d_618_conv2d_readvariableop_resource:@F
8batch_instance_normalization_510_readvariableop_resource:L
>batch_instance_normalization_510_mul_4_readvariableop_resource:L
>batch_instance_normalization_510_add_3_readvariableop_resource:C
)conv2d_619_conv2d_readvariableop_resource:
identityĒ/batch_instance_normalization_500/ReadVariableOpĒ1batch_instance_normalization_500/ReadVariableOp_1Ē5batch_instance_normalization_500/add_3/ReadVariableOpĒ5batch_instance_normalization_500/mul_4/ReadVariableOpĒ/batch_instance_normalization_501/ReadVariableOpĒ1batch_instance_normalization_501/ReadVariableOp_1Ē5batch_instance_normalization_501/add_3/ReadVariableOpĒ5batch_instance_normalization_501/mul_4/ReadVariableOpĒ/batch_instance_normalization_502/ReadVariableOpĒ1batch_instance_normalization_502/ReadVariableOp_1Ē5batch_instance_normalization_502/add_3/ReadVariableOpĒ5batch_instance_normalization_502/mul_4/ReadVariableOpĒ/batch_instance_normalization_503/ReadVariableOpĒ1batch_instance_normalization_503/ReadVariableOp_1Ē5batch_instance_normalization_503/add_3/ReadVariableOpĒ5batch_instance_normalization_503/mul_4/ReadVariableOpĒ/batch_instance_normalization_504/ReadVariableOpĒ1batch_instance_normalization_504/ReadVariableOp_1Ē5batch_instance_normalization_504/add_3/ReadVariableOpĒ5batch_instance_normalization_504/mul_4/ReadVariableOpĒ/batch_instance_normalization_505/ReadVariableOpĒ1batch_instance_normalization_505/ReadVariableOp_1Ē5batch_instance_normalization_505/add_3/ReadVariableOpĒ5batch_instance_normalization_505/mul_4/ReadVariableOpĒ/batch_instance_normalization_506/ReadVariableOpĒ1batch_instance_normalization_506/ReadVariableOp_1Ē5batch_instance_normalization_506/add_3/ReadVariableOpĒ5batch_instance_normalization_506/mul_4/ReadVariableOpĒ/batch_instance_normalization_507/ReadVariableOpĒ1batch_instance_normalization_507/ReadVariableOp_1Ē5batch_instance_normalization_507/add_3/ReadVariableOpĒ5batch_instance_normalization_507/mul_4/ReadVariableOpĒ/batch_instance_normalization_508/ReadVariableOpĒ1batch_instance_normalization_508/ReadVariableOp_1Ē5batch_instance_normalization_508/add_3/ReadVariableOpĒ5batch_instance_normalization_508/mul_4/ReadVariableOpĒ/batch_instance_normalization_509/ReadVariableOpĒ1batch_instance_normalization_509/ReadVariableOp_1Ē5batch_instance_normalization_509/add_3/ReadVariableOpĒ5batch_instance_normalization_509/mul_4/ReadVariableOpĒ/batch_instance_normalization_510/ReadVariableOpĒ1batch_instance_normalization_510/ReadVariableOp_1Ē5batch_instance_normalization_510/add_3/ReadVariableOpĒ5batch_instance_normalization_510/mul_4/ReadVariableOpĒ conv2d_606/Conv2D/ReadVariableOpĒ conv2d_607/Conv2D/ReadVariableOpĒ conv2d_608/Conv2D/ReadVariableOpĒ conv2d_609/Conv2D/ReadVariableOpĒ conv2d_610/Conv2D/ReadVariableOpĒ conv2d_611/Conv2D/ReadVariableOpĒ conv2d_612/Conv2D/ReadVariableOpĒ conv2d_613/Conv2D/ReadVariableOpĒ conv2d_614/Conv2D/ReadVariableOpĒ conv2d_615/Conv2D/ReadVariableOpĒ conv2d_616/Conv2D/ReadVariableOpĒ conv2d_617/Conv2D/ReadVariableOpĒ conv2d_618/Conv2D/ReadVariableOpĒ conv2d_619/Conv2D/ReadVariableOpĒ4conv2d_transpose_100/conv2d_transpose/ReadVariableOpĒ4conv2d_transpose_101/conv2d_transpose/ReadVariableOpY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_606/Conv2D/ReadVariableOpReadVariableOp)conv2d_606_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Æ
conv2d_606/Conv2DConv2Dconcatenate/concat:output:0(conv2d_606/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

 conv2d_607/Conv2D/ReadVariableOpReadVariableOp)conv2d_607_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Å
conv2d_607/Conv2DConv2Dconv2d_606/Conv2D:output:0(conv2d_607/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
e
	LeakyRelu	LeakyReluconv2d_607/Conv2D:output:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Š
max_pooling2d_100/MaxPoolMaxPoolLeakyRelu:activations:0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@@*
ksize
*
paddingVALID*
strides

 conv2d_608/Conv2D/ReadVariableOpReadVariableOp)conv2d_608_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ė
conv2d_608/Conv2DConv2D"max_pooling2d_100/MaxPool:output:0(conv2d_608/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

?batch_instance_normalization_500/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Þ
-batch_instance_normalization_500/moments/meanMeanconv2d_608/Conv2D:output:0Hbatch_instance_normalization_500/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_500/moments/StopGradientStopGradient6batch_instance_normalization_500/moments/mean:output:0*
T0*'
_output_shapes
:æ
:batch_instance_normalization_500/moments/SquaredDifferenceSquaredDifferenceconv2d_608/Conv2D:output:0>batch_instance_normalization_500/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Cbatch_instance_normalization_500/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_500/moments/varianceMean>batch_instance_normalization_500/moments/SquaredDifference:z:0Lbatch_instance_normalization_500/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(š
$batch_instance_normalization_500/subSubconv2d_608/Conv2D:output:06batch_instance_normalization_500/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@k
&batch_instance_normalization_500/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_500/addAddV2:batch_instance_normalization_500/moments/variance:output:0/batch_instance_normalization_500/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_500/RsqrtRsqrt(batch_instance_normalization_500/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_500/mulMul(batch_instance_normalization_500/sub:z:0*batch_instance_normalization_500/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Abatch_instance_normalization_500/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ë
/batch_instance_normalization_500/moments_1/meanMeanconv2d_608/Conv2D:output:0Jbatch_instance_normalization_500/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_500/moments_1/StopGradientStopGradient8batch_instance_normalization_500/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸę
<batch_instance_normalization_500/moments_1/SquaredDifferenceSquaredDifferenceconv2d_608/Conv2D:output:0@batch_instance_normalization_500/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Ebatch_instance_normalization_500/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_500/moments_1/varianceMean@batch_instance_normalization_500/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_500/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ū
&batch_instance_normalization_500/sub_1Subconv2d_608/Conv2D:output:08batch_instance_normalization_500/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@m
(batch_instance_normalization_500/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_500/add_1AddV2<batch_instance_normalization_500/moments_1/variance:output:01batch_instance_normalization_500/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_500/Rsqrt_1Rsqrt*batch_instance_normalization_500/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_500/mul_1Mul*batch_instance_normalization_500/sub_1:z:0,batch_instance_normalization_500/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
/batch_instance_normalization_500/ReadVariableOpReadVariableOp8batch_instance_normalization_500_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_500/mul_2Mul7batch_instance_normalization_500/ReadVariableOp:value:0(batch_instance_normalization_500/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@§
1batch_instance_normalization_500/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_500_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_500/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_500/sub_2Sub1batch_instance_normalization_500/sub_2/x:output:09batch_instance_normalization_500/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_500/mul_3Mul*batch_instance_normalization_500/sub_2:z:0*batch_instance_normalization_500/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Â
&batch_instance_normalization_500/add_2AddV2*batch_instance_normalization_500/mul_2:z:0*batch_instance_normalization_500/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_500/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_500_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_500/mul_4Mul*batch_instance_normalization_500/add_2:z:0=batch_instance_normalization_500/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_500/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_500_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_500/add_3AddV2*batch_instance_normalization_500/mul_4:z:0=batch_instance_normalization_500/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@v
LeakyRelu_1	LeakyRelu*batch_instance_normalization_500/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 conv2d_609/Conv2D/ReadVariableOpReadVariableOp)conv2d_609_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ã
conv2d_609/Conv2DConv2DLeakyRelu_1:activations:0(conv2d_609/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

?batch_instance_normalization_501/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Þ
-batch_instance_normalization_501/moments/meanMeanconv2d_609/Conv2D:output:0Hbatch_instance_normalization_501/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_501/moments/StopGradientStopGradient6batch_instance_normalization_501/moments/mean:output:0*
T0*'
_output_shapes
:æ
:batch_instance_normalization_501/moments/SquaredDifferenceSquaredDifferenceconv2d_609/Conv2D:output:0>batch_instance_normalization_501/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Cbatch_instance_normalization_501/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_501/moments/varianceMean>batch_instance_normalization_501/moments/SquaredDifference:z:0Lbatch_instance_normalization_501/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(š
$batch_instance_normalization_501/subSubconv2d_609/Conv2D:output:06batch_instance_normalization_501/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@k
&batch_instance_normalization_501/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_501/addAddV2:batch_instance_normalization_501/moments/variance:output:0/batch_instance_normalization_501/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_501/RsqrtRsqrt(batch_instance_normalization_501/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_501/mulMul(batch_instance_normalization_501/sub:z:0*batch_instance_normalization_501/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Abatch_instance_normalization_501/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ë
/batch_instance_normalization_501/moments_1/meanMeanconv2d_609/Conv2D:output:0Jbatch_instance_normalization_501/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_501/moments_1/StopGradientStopGradient8batch_instance_normalization_501/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸę
<batch_instance_normalization_501/moments_1/SquaredDifferenceSquaredDifferenceconv2d_609/Conv2D:output:0@batch_instance_normalization_501/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Ebatch_instance_normalization_501/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_501/moments_1/varianceMean@batch_instance_normalization_501/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_501/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ū
&batch_instance_normalization_501/sub_1Subconv2d_609/Conv2D:output:08batch_instance_normalization_501/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@m
(batch_instance_normalization_501/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_501/add_1AddV2<batch_instance_normalization_501/moments_1/variance:output:01batch_instance_normalization_501/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_501/Rsqrt_1Rsqrt*batch_instance_normalization_501/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_501/mul_1Mul*batch_instance_normalization_501/sub_1:z:0,batch_instance_normalization_501/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
/batch_instance_normalization_501/ReadVariableOpReadVariableOp8batch_instance_normalization_501_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_501/mul_2Mul7batch_instance_normalization_501/ReadVariableOp:value:0(batch_instance_normalization_501/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@§
1batch_instance_normalization_501/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_501_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_501/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_501/sub_2Sub1batch_instance_normalization_501/sub_2/x:output:09batch_instance_normalization_501/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_501/mul_3Mul*batch_instance_normalization_501/sub_2:z:0*batch_instance_normalization_501/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Â
&batch_instance_normalization_501/add_2AddV2*batch_instance_normalization_501/mul_2:z:0*batch_instance_normalization_501/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_501/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_501_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_501/mul_4Mul*batch_instance_normalization_501/add_2:z:0=batch_instance_normalization_501/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_501/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_501_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_501/add_3AddV2*batch_instance_normalization_501/mul_4:z:0=batch_instance_normalization_501/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@v
LeakyRelu_2	LeakyRelu*batch_instance_normalization_501/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@­
max_pooling2d_101/MaxPoolMaxPoolLeakyRelu_2:activations:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  *
ksize
*
paddingVALID*
strides
p
conv2d_610/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      w
conv2d_610/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            x
conv2d_610/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
>conv2d_610/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
;conv2d_610/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
8conv2d_610/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                }
,conv2d_610/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)conv2d_610/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ü
 conv2d_610/Conv2D/SpaceToBatchNDSpaceToBatchND"max_pooling2d_101/MaxPool:output:05conv2d_610/Conv2D/SpaceToBatchND/block_shape:output:02conv2d_610/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_610/Conv2D/ReadVariableOpReadVariableOp)conv2d_610_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_610/Conv2DConv2D)conv2d_610/Conv2D/SpaceToBatchND:output:0(conv2d_610/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
}
,conv2d_610/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
&conv2d_610/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                ņ
 conv2d_610/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_610/Conv2D:output:05conv2d_610/Conv2D/BatchToSpaceND/block_shape:output:0/conv2d_610/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
?batch_instance_normalization_502/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          í
-batch_instance_normalization_502/moments/meanMean)conv2d_610/Conv2D/BatchToSpaceND:output:0Hbatch_instance_normalization_502/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_502/moments/StopGradientStopGradient6batch_instance_normalization_502/moments/mean:output:0*
T0*'
_output_shapes
:õ
:batch_instance_normalization_502/moments/SquaredDifferenceSquaredDifference)conv2d_610/Conv2D/BatchToSpaceND:output:0>batch_instance_normalization_502/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Cbatch_instance_normalization_502/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_502/moments/varianceMean>batch_instance_normalization_502/moments/SquaredDifference:z:0Lbatch_instance_normalization_502/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(É
$batch_instance_normalization_502/subSub)conv2d_610/Conv2D/BatchToSpaceND:output:06batch_instance_normalization_502/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  k
&batch_instance_normalization_502/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_502/addAddV2:batch_instance_normalization_502/moments/variance:output:0/batch_instance_normalization_502/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_502/RsqrtRsqrt(batch_instance_normalization_502/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_502/mulMul(batch_instance_normalization_502/sub:z:0*batch_instance_normalization_502/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Abatch_instance_normalization_502/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ú
/batch_instance_normalization_502/moments_1/meanMean)conv2d_610/Conv2D/BatchToSpaceND:output:0Jbatch_instance_normalization_502/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_502/moments_1/StopGradientStopGradient8batch_instance_normalization_502/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸų
<batch_instance_normalization_502/moments_1/SquaredDifferenceSquaredDifference)conv2d_610/Conv2D/BatchToSpaceND:output:0@batch_instance_normalization_502/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Ebatch_instance_normalization_502/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_502/moments_1/varianceMean@batch_instance_normalization_502/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_502/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Í
&batch_instance_normalization_502/sub_1Sub)conv2d_610/Conv2D/BatchToSpaceND:output:08batch_instance_normalization_502/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  m
(batch_instance_normalization_502/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_502/add_1AddV2<batch_instance_normalization_502/moments_1/variance:output:01batch_instance_normalization_502/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_502/Rsqrt_1Rsqrt*batch_instance_normalization_502/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_502/mul_1Mul*batch_instance_normalization_502/sub_1:z:0,batch_instance_normalization_502/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ĩ
/batch_instance_normalization_502/ReadVariableOpReadVariableOp8batch_instance_normalization_502_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_502/mul_2Mul7batch_instance_normalization_502/ReadVariableOp:value:0(batch_instance_normalization_502/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  §
1batch_instance_normalization_502/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_502_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_502/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_502/sub_2Sub1batch_instance_normalization_502/sub_2/x:output:09batch_instance_normalization_502/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_502/mul_3Mul*batch_instance_normalization_502/sub_2:z:0*batch_instance_normalization_502/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Â
&batch_instance_normalization_502/add_2AddV2*batch_instance_normalization_502/mul_2:z:0*batch_instance_normalization_502/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_502/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_502_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_502/mul_4Mul*batch_instance_normalization_502/add_2:z:0=batch_instance_normalization_502/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_502/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_502_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_502/add_3AddV2*batch_instance_normalization_502/mul_4:z:0=batch_instance_normalization_502/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
LeakyRelu_3	LeakyRelu*batch_instance_normalization_502/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  p
conv2d_611/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      w
conv2d_611/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            x
conv2d_611/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
>conv2d_611/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
;conv2d_611/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
8conv2d_611/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                }
,conv2d_611/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)conv2d_611/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ó
 conv2d_611/Conv2D/SpaceToBatchNDSpaceToBatchNDLeakyRelu_3:activations:05conv2d_611/Conv2D/SpaceToBatchND/block_shape:output:02conv2d_611/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ


 conv2d_611/Conv2D/ReadVariableOpReadVariableOp)conv2d_611_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_611/Conv2DConv2D)conv2d_611/Conv2D/SpaceToBatchND:output:0(conv2d_611/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
}
,conv2d_611/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
&conv2d_611/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                ņ
 conv2d_611/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_611/Conv2D:output:05conv2d_611/Conv2D/BatchToSpaceND/block_shape:output:0/conv2d_611/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
?batch_instance_normalization_503/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          í
-batch_instance_normalization_503/moments/meanMean)conv2d_611/Conv2D/BatchToSpaceND:output:0Hbatch_instance_normalization_503/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_503/moments/StopGradientStopGradient6batch_instance_normalization_503/moments/mean:output:0*
T0*'
_output_shapes
:õ
:batch_instance_normalization_503/moments/SquaredDifferenceSquaredDifference)conv2d_611/Conv2D/BatchToSpaceND:output:0>batch_instance_normalization_503/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Cbatch_instance_normalization_503/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_503/moments/varianceMean>batch_instance_normalization_503/moments/SquaredDifference:z:0Lbatch_instance_normalization_503/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(É
$batch_instance_normalization_503/subSub)conv2d_611/Conv2D/BatchToSpaceND:output:06batch_instance_normalization_503/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  k
&batch_instance_normalization_503/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_503/addAddV2:batch_instance_normalization_503/moments/variance:output:0/batch_instance_normalization_503/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_503/RsqrtRsqrt(batch_instance_normalization_503/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_503/mulMul(batch_instance_normalization_503/sub:z:0*batch_instance_normalization_503/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Abatch_instance_normalization_503/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ú
/batch_instance_normalization_503/moments_1/meanMean)conv2d_611/Conv2D/BatchToSpaceND:output:0Jbatch_instance_normalization_503/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_503/moments_1/StopGradientStopGradient8batch_instance_normalization_503/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸų
<batch_instance_normalization_503/moments_1/SquaredDifferenceSquaredDifference)conv2d_611/Conv2D/BatchToSpaceND:output:0@batch_instance_normalization_503/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Ebatch_instance_normalization_503/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_503/moments_1/varianceMean@batch_instance_normalization_503/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_503/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Í
&batch_instance_normalization_503/sub_1Sub)conv2d_611/Conv2D/BatchToSpaceND:output:08batch_instance_normalization_503/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  m
(batch_instance_normalization_503/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_503/add_1AddV2<batch_instance_normalization_503/moments_1/variance:output:01batch_instance_normalization_503/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_503/Rsqrt_1Rsqrt*batch_instance_normalization_503/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_503/mul_1Mul*batch_instance_normalization_503/sub_1:z:0,batch_instance_normalization_503/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ĩ
/batch_instance_normalization_503/ReadVariableOpReadVariableOp8batch_instance_normalization_503_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_503/mul_2Mul7batch_instance_normalization_503/ReadVariableOp:value:0(batch_instance_normalization_503/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  §
1batch_instance_normalization_503/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_503_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_503/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_503/sub_2Sub1batch_instance_normalization_503/sub_2/x:output:09batch_instance_normalization_503/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_503/mul_3Mul*batch_instance_normalization_503/sub_2:z:0*batch_instance_normalization_503/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Â
&batch_instance_normalization_503/add_2AddV2*batch_instance_normalization_503/mul_2:z:0*batch_instance_normalization_503/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_503/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_503_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_503/mul_4Mul*batch_instance_normalization_503/add_2:z:0=batch_instance_normalization_503/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_503/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_503_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_503/add_3AddV2*batch_instance_normalization_503/mul_4:z:0=batch_instance_normalization_503/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
LeakyRelu_4	LeakyRelu*batch_instance_normalization_503/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  p
conv2d_612/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      w
conv2d_612/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            x
conv2d_612/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
>conv2d_612/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
;conv2d_612/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
8conv2d_612/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                }
,conv2d_612/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)conv2d_612/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ó
 conv2d_612/Conv2D/SpaceToBatchNDSpaceToBatchNDLeakyRelu_4:activations:05conv2d_612/Conv2D/SpaceToBatchND/block_shape:output:02conv2d_612/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_612/Conv2D/ReadVariableOpReadVariableOp)conv2d_612_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_612/Conv2DConv2D)conv2d_612/Conv2D/SpaceToBatchND:output:0(conv2d_612/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
}
,conv2d_612/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
&conv2d_612/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                ņ
 conv2d_612/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_612/Conv2D:output:05conv2d_612/Conv2D/BatchToSpaceND/block_shape:output:0/conv2d_612/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
?batch_instance_normalization_504/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          í
-batch_instance_normalization_504/moments/meanMean)conv2d_612/Conv2D/BatchToSpaceND:output:0Hbatch_instance_normalization_504/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_504/moments/StopGradientStopGradient6batch_instance_normalization_504/moments/mean:output:0*
T0*'
_output_shapes
:õ
:batch_instance_normalization_504/moments/SquaredDifferenceSquaredDifference)conv2d_612/Conv2D/BatchToSpaceND:output:0>batch_instance_normalization_504/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Cbatch_instance_normalization_504/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_504/moments/varianceMean>batch_instance_normalization_504/moments/SquaredDifference:z:0Lbatch_instance_normalization_504/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(É
$batch_instance_normalization_504/subSub)conv2d_612/Conv2D/BatchToSpaceND:output:06batch_instance_normalization_504/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  k
&batch_instance_normalization_504/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_504/addAddV2:batch_instance_normalization_504/moments/variance:output:0/batch_instance_normalization_504/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_504/RsqrtRsqrt(batch_instance_normalization_504/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_504/mulMul(batch_instance_normalization_504/sub:z:0*batch_instance_normalization_504/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Abatch_instance_normalization_504/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ú
/batch_instance_normalization_504/moments_1/meanMean)conv2d_612/Conv2D/BatchToSpaceND:output:0Jbatch_instance_normalization_504/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_504/moments_1/StopGradientStopGradient8batch_instance_normalization_504/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸų
<batch_instance_normalization_504/moments_1/SquaredDifferenceSquaredDifference)conv2d_612/Conv2D/BatchToSpaceND:output:0@batch_instance_normalization_504/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Ebatch_instance_normalization_504/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_504/moments_1/varianceMean@batch_instance_normalization_504/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_504/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Í
&batch_instance_normalization_504/sub_1Sub)conv2d_612/Conv2D/BatchToSpaceND:output:08batch_instance_normalization_504/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  m
(batch_instance_normalization_504/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_504/add_1AddV2<batch_instance_normalization_504/moments_1/variance:output:01batch_instance_normalization_504/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_504/Rsqrt_1Rsqrt*batch_instance_normalization_504/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_504/mul_1Mul*batch_instance_normalization_504/sub_1:z:0,batch_instance_normalization_504/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ĩ
/batch_instance_normalization_504/ReadVariableOpReadVariableOp8batch_instance_normalization_504_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_504/mul_2Mul7batch_instance_normalization_504/ReadVariableOp:value:0(batch_instance_normalization_504/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  §
1batch_instance_normalization_504/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_504_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_504/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_504/sub_2Sub1batch_instance_normalization_504/sub_2/x:output:09batch_instance_normalization_504/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_504/mul_3Mul*batch_instance_normalization_504/sub_2:z:0*batch_instance_normalization_504/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Â
&batch_instance_normalization_504/add_2AddV2*batch_instance_normalization_504/mul_2:z:0*batch_instance_normalization_504/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_504/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_504_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_504/mul_4Mul*batch_instance_normalization_504/add_2:z:0=batch_instance_normalization_504/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_504/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_504_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_504/add_3AddV2*batch_instance_normalization_504/mul_4:z:0=batch_instance_normalization_504/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
LeakyRelu_5	LeakyRelu*batch_instance_normalization_504/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  p
conv2d_613/Conv2D/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      w
conv2d_613/Conv2D/filter_shapeConst*
_output_shapes
:*
dtype0*%
valueB"            x
conv2d_613/Conv2D/stackConst*
_output_shapes

:*
dtype0*)
value B"            
>conv2d_613/Conv2D/required_space_to_batch_paddings/input_shapeConst*
_output_shapes
:*
dtype0*
valueB"        
;conv2d_613/Conv2D/required_space_to_batch_paddings/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            
8conv2d_613/Conv2D/required_space_to_batch_paddings/cropsConst*
_output_shapes

:*
dtype0*)
value B"                }
,conv2d_613/Conv2D/SpaceToBatchND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
)conv2d_613/Conv2D/SpaceToBatchND/paddingsConst*
_output_shapes

:*
dtype0*)
value B"            ó
 conv2d_613/Conv2D/SpaceToBatchNDSpaceToBatchNDLeakyRelu_5:activations:05conv2d_613/Conv2D/SpaceToBatchND/block_shape:output:02conv2d_613/Conv2D/SpaceToBatchND/paddings:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_613/Conv2D/ReadVariableOpReadVariableOp)conv2d_613_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_613/Conv2DConv2D)conv2d_613/Conv2D/SpaceToBatchND:output:0(conv2d_613/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingVALID*
strides
}
,conv2d_613/Conv2D/BatchToSpaceND/block_shapeConst*
_output_shapes
:*
dtype0*
valueB"      
&conv2d_613/Conv2D/BatchToSpaceND/cropsConst*
_output_shapes

:*
dtype0*)
value B"                ņ
 conv2d_613/Conv2D/BatchToSpaceNDBatchToSpaceNDconv2d_613/Conv2D:output:05conv2d_613/Conv2D/BatchToSpaceND/block_shape:output:0/conv2d_613/Conv2D/BatchToSpaceND/crops:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
?batch_instance_normalization_505/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          í
-batch_instance_normalization_505/moments/meanMean)conv2d_613/Conv2D/BatchToSpaceND:output:0Hbatch_instance_normalization_505/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_505/moments/StopGradientStopGradient6batch_instance_normalization_505/moments/mean:output:0*
T0*'
_output_shapes
:õ
:batch_instance_normalization_505/moments/SquaredDifferenceSquaredDifference)conv2d_613/Conv2D/BatchToSpaceND:output:0>batch_instance_normalization_505/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Cbatch_instance_normalization_505/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_505/moments/varianceMean>batch_instance_normalization_505/moments/SquaredDifference:z:0Lbatch_instance_normalization_505/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(É
$batch_instance_normalization_505/subSub)conv2d_613/Conv2D/BatchToSpaceND:output:06batch_instance_normalization_505/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  k
&batch_instance_normalization_505/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_505/addAddV2:batch_instance_normalization_505/moments/variance:output:0/batch_instance_normalization_505/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_505/RsqrtRsqrt(batch_instance_normalization_505/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_505/mulMul(batch_instance_normalization_505/sub:z:0*batch_instance_normalization_505/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Abatch_instance_normalization_505/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ú
/batch_instance_normalization_505/moments_1/meanMean)conv2d_613/Conv2D/BatchToSpaceND:output:0Jbatch_instance_normalization_505/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_505/moments_1/StopGradientStopGradient8batch_instance_normalization_505/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸų
<batch_instance_normalization_505/moments_1/SquaredDifferenceSquaredDifference)conv2d_613/Conv2D/BatchToSpaceND:output:0@batch_instance_normalization_505/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  
Ebatch_instance_normalization_505/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_505/moments_1/varianceMean@batch_instance_normalization_505/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_505/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(Í
&batch_instance_normalization_505/sub_1Sub)conv2d_613/Conv2D/BatchToSpaceND:output:08batch_instance_normalization_505/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  m
(batch_instance_normalization_505/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_505/add_1AddV2<batch_instance_normalization_505/moments_1/variance:output:01batch_instance_normalization_505/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_505/Rsqrt_1Rsqrt*batch_instance_normalization_505/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_505/mul_1Mul*batch_instance_normalization_505/sub_1:z:0,batch_instance_normalization_505/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Ĩ
/batch_instance_normalization_505/ReadVariableOpReadVariableOp8batch_instance_normalization_505_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_505/mul_2Mul7batch_instance_normalization_505/ReadVariableOp:value:0(batch_instance_normalization_505/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  §
1batch_instance_normalization_505/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_505_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_505/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_505/sub_2Sub1batch_instance_normalization_505/sub_2/x:output:09batch_instance_normalization_505/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_505/mul_3Mul*batch_instance_normalization_505/sub_2:z:0*batch_instance_normalization_505/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  Â
&batch_instance_normalization_505/add_2AddV2*batch_instance_normalization_505/mul_2:z:0*batch_instance_normalization_505/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_505/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_505_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_505/mul_4Mul*batch_instance_normalization_505/add_2:z:0=batch_instance_normalization_505/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  ą
5batch_instance_normalization_505/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_505_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_505/add_3AddV2*batch_instance_normalization_505/mul_4:z:0=batch_instance_normalization_505/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  v
LeakyRelu_6	LeakyRelu*batch_instance_normalization_505/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ  c
conv2d_transpose_100/ShapeShapeLeakyRelu_6:activations:0*
T0*
_output_shapes
:r
(conv2d_transpose_100/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_100/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_100/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:š
"conv2d_transpose_100/strided_sliceStridedSlice#conv2d_transpose_100/Shape:output:01conv2d_transpose_100/strided_slice/stack:output:03conv2d_transpose_100/strided_slice/stack_1:output:03conv2d_transpose_100/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_100/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@^
conv2d_transpose_100/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@_
conv2d_transpose_100/stack/3Const*
_output_shapes
: *
dtype0*
value
B :ō
conv2d_transpose_100/stackPack+conv2d_transpose_100/strided_slice:output:0%conv2d_transpose_100/stack/1:output:0%conv2d_transpose_100/stack/2:output:0%conv2d_transpose_100/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_100/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_100/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_100/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Â
$conv2d_transpose_100/strided_slice_1StridedSlice#conv2d_transpose_100/stack:output:03conv2d_transpose_100/strided_slice_1/stack:output:05conv2d_transpose_100/strided_slice_1/stack_1:output:05conv2d_transpose_100/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskž
4conv2d_transpose_100/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_100_conv2d_transpose_readvariableop_resource*(
_output_shapes
:*
dtype0
%conv2d_transpose_100/conv2d_transposeConv2DBackpropInput#conv2d_transpose_100/stack:output:0<conv2d_transpose_100/conv2d_transpose/ReadVariableOp:value:0LeakyRelu_6:activations:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides
[
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ó
concatenate_1/concatConcatV2LeakyRelu_2:activations:0.conv2d_transpose_100/conv2d_transpose:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 conv2d_614/Conv2D/ReadVariableOpReadVariableOp)conv2d_614_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Į
conv2d_614/Conv2DConv2Dconcatenate_1/concat:output:0(conv2d_614/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

?batch_instance_normalization_506/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Þ
-batch_instance_normalization_506/moments/meanMeanconv2d_614/Conv2D:output:0Hbatch_instance_normalization_506/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_506/moments/StopGradientStopGradient6batch_instance_normalization_506/moments/mean:output:0*
T0*'
_output_shapes
:æ
:batch_instance_normalization_506/moments/SquaredDifferenceSquaredDifferenceconv2d_614/Conv2D:output:0>batch_instance_normalization_506/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Cbatch_instance_normalization_506/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_506/moments/varianceMean>batch_instance_normalization_506/moments/SquaredDifference:z:0Lbatch_instance_normalization_506/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(š
$batch_instance_normalization_506/subSubconv2d_614/Conv2D:output:06batch_instance_normalization_506/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@k
&batch_instance_normalization_506/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_506/addAddV2:batch_instance_normalization_506/moments/variance:output:0/batch_instance_normalization_506/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_506/RsqrtRsqrt(batch_instance_normalization_506/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_506/mulMul(batch_instance_normalization_506/sub:z:0*batch_instance_normalization_506/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Abatch_instance_normalization_506/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ë
/batch_instance_normalization_506/moments_1/meanMeanconv2d_614/Conv2D:output:0Jbatch_instance_normalization_506/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_506/moments_1/StopGradientStopGradient8batch_instance_normalization_506/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸę
<batch_instance_normalization_506/moments_1/SquaredDifferenceSquaredDifferenceconv2d_614/Conv2D:output:0@batch_instance_normalization_506/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Ebatch_instance_normalization_506/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_506/moments_1/varianceMean@batch_instance_normalization_506/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_506/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ū
&batch_instance_normalization_506/sub_1Subconv2d_614/Conv2D:output:08batch_instance_normalization_506/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@m
(batch_instance_normalization_506/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_506/add_1AddV2<batch_instance_normalization_506/moments_1/variance:output:01batch_instance_normalization_506/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_506/Rsqrt_1Rsqrt*batch_instance_normalization_506/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_506/mul_1Mul*batch_instance_normalization_506/sub_1:z:0,batch_instance_normalization_506/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
/batch_instance_normalization_506/ReadVariableOpReadVariableOp8batch_instance_normalization_506_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_506/mul_2Mul7batch_instance_normalization_506/ReadVariableOp:value:0(batch_instance_normalization_506/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@§
1batch_instance_normalization_506/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_506_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_506/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_506/sub_2Sub1batch_instance_normalization_506/sub_2/x:output:09batch_instance_normalization_506/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_506/mul_3Mul*batch_instance_normalization_506/sub_2:z:0*batch_instance_normalization_506/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Â
&batch_instance_normalization_506/add_2AddV2*batch_instance_normalization_506/mul_2:z:0*batch_instance_normalization_506/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_506/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_506_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_506/mul_4Mul*batch_instance_normalization_506/add_2:z:0=batch_instance_normalization_506/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_506/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_506_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_506/add_3AddV2*batch_instance_normalization_506/mul_4:z:0=batch_instance_normalization_506/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@v
LeakyRelu_7	LeakyRelu*batch_instance_normalization_506/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
 conv2d_615/Conv2D/ReadVariableOpReadVariableOp)conv2d_615_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ã
conv2d_615/Conv2DConv2DLeakyRelu_7:activations:0(conv2d_615/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*
paddingSAME*
strides

?batch_instance_normalization_507/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Þ
-batch_instance_normalization_507/moments/meanMeanconv2d_615/Conv2D:output:0Hbatch_instance_normalization_507/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(Ŋ
5batch_instance_normalization_507/moments/StopGradientStopGradient6batch_instance_normalization_507/moments/mean:output:0*
T0*'
_output_shapes
:æ
:batch_instance_normalization_507/moments/SquaredDifferenceSquaredDifferenceconv2d_615/Conv2D:output:0>batch_instance_normalization_507/moments/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Cbatch_instance_normalization_507/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_507/moments/varianceMean>batch_instance_normalization_507/moments/SquaredDifference:z:0Lbatch_instance_normalization_507/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(š
$batch_instance_normalization_507/subSubconv2d_615/Conv2D:output:06batch_instance_normalization_507/moments/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@k
&batch_instance_normalization_507/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ė
$batch_instance_normalization_507/addAddV2:batch_instance_normalization_507/moments/variance:output:0/batch_instance_normalization_507/add/y:output:0*
T0*'
_output_shapes
:
&batch_instance_normalization_507/RsqrtRsqrt(batch_instance_normalization_507/add:z:0*
T0*'
_output_shapes
:ž
$batch_instance_normalization_507/mulMul(batch_instance_normalization_507/sub:z:0*batch_instance_normalization_507/Rsqrt:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Abatch_instance_normalization_507/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ë
/batch_instance_normalization_507/moments_1/meanMeanconv2d_615/Conv2D:output:0Jbatch_instance_normalization_507/moments_1/mean/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ž
7batch_instance_normalization_507/moments_1/StopGradientStopGradient8batch_instance_normalization_507/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸę
<batch_instance_normalization_507/moments_1/SquaredDifferenceSquaredDifferenceconv2d_615/Conv2D:output:0@batch_instance_normalization_507/moments_1/StopGradient:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@
Ebatch_instance_normalization_507/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_507/moments_1/varianceMean@batch_instance_normalization_507/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_507/moments_1/variance/reduction_indices:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ū
&batch_instance_normalization_507/sub_1Subconv2d_615/Conv2D:output:08batch_instance_normalization_507/moments_1/mean:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@m
(batch_instance_normalization_507/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Û
&batch_instance_normalization_507/add_1AddV2<batch_instance_normalization_507/moments_1/variance:output:01batch_instance_normalization_507/add_1/y:output:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_507/Rsqrt_1Rsqrt*batch_instance_normalization_507/add_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸÂ
&batch_instance_normalization_507/mul_1Mul*batch_instance_normalization_507/sub_1:z:0,batch_instance_normalization_507/Rsqrt_1:y:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Ĩ
/batch_instance_normalization_507/ReadVariableOpReadVariableOp8batch_instance_normalization_507_readvariableop_resource*
_output_shapes	
:*
dtype0Ë
&batch_instance_normalization_507/mul_2Mul7batch_instance_normalization_507/ReadVariableOp:value:0(batch_instance_normalization_507/mul:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@§
1batch_instance_normalization_507/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_507_readvariableop_resource*
_output_shapes	
:*
dtype0m
(batch_instance_normalization_507/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Á
&batch_instance_normalization_507/sub_2Sub1batch_instance_normalization_507/sub_2/x:output:09batch_instance_normalization_507/ReadVariableOp_1:value:0*
T0*
_output_shapes	
:Ā
&batch_instance_normalization_507/mul_3Mul*batch_instance_normalization_507/sub_2:z:0*batch_instance_normalization_507/mul_1:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@Â
&batch_instance_normalization_507/add_2AddV2*batch_instance_normalization_507/mul_2:z:0*batch_instance_normalization_507/mul_3:z:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_507/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_507_mul_4_readvariableop_resource*
_output_shapes	
:*
dtype0Ó
&batch_instance_normalization_507/mul_4Mul*batch_instance_normalization_507/add_2:z:0=batch_instance_normalization_507/mul_4/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@ą
5batch_instance_normalization_507/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_507_add_3_readvariableop_resource*
_output_shapes	
:*
dtype0Õ
&batch_instance_normalization_507/add_3AddV2*batch_instance_normalization_507/mul_4:z:0=batch_instance_normalization_507/add_3/ReadVariableOp:value:0*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@v
LeakyRelu_8	LeakyRelu*batch_instance_normalization_507/add_3:z:0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@c
conv2d_transpose_101/ShapeShapeLeakyRelu_8:activations:0*
T0*
_output_shapes
:r
(conv2d_transpose_101/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_101/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_101/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:š
"conv2d_transpose_101/strided_sliceStridedSlice#conv2d_transpose_101/Shape:output:01conv2d_transpose_101/strided_slice/stack:output:03conv2d_transpose_101/strided_slice/stack_1:output:03conv2d_transpose_101/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
conv2d_transpose_101/stack/1Const*
_output_shapes
: *
dtype0*
value
B :_
conv2d_transpose_101/stack/2Const*
_output_shapes
: *
dtype0*
value
B :^
conv2d_transpose_101/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@ō
conv2d_transpose_101/stackPack+conv2d_transpose_101/strided_slice:output:0%conv2d_transpose_101/stack/1:output:0%conv2d_transpose_101/stack/2:output:0%conv2d_transpose_101/stack/3:output:0*
N*
T0*
_output_shapes
:t
*conv2d_transpose_101/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,conv2d_transpose_101/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,conv2d_transpose_101/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Â
$conv2d_transpose_101/strided_slice_1StridedSlice#conv2d_transpose_101/stack:output:03conv2d_transpose_101/strided_slice_1/stack:output:05conv2d_transpose_101/strided_slice_1/stack_1:output:05conv2d_transpose_101/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskŧ
4conv2d_transpose_101/conv2d_transpose/ReadVariableOpReadVariableOp=conv2d_transpose_101_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@*
dtype0
%conv2d_transpose_101/conv2d_transposeConv2DBackpropInput#conv2d_transpose_101/stack:output:0<conv2d_transpose_101/conv2d_transpose/ReadVariableOp:value:0LeakyRelu_8:activations:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
[
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ó
concatenate_2/concatConcatV2LeakyRelu:activations:0.conv2d_transpose_101/conv2d_transpose:output:0"concatenate_2/concat/axis:output:0*
N*
T0*2
_output_shapes 
:ĸĸĸĸĸĸĸĸĸ
 conv2d_616/Conv2D/ReadVariableOpReadVariableOp)conv2d_616_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Č
conv2d_616/Conv2DConv2Dconcatenate_2/concat:output:0(conv2d_616/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

?batch_instance_normalization_508/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ý
-batch_instance_normalization_508/moments/meanMeanconv2d_616/Conv2D:output:0Hbatch_instance_normalization_508/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(Ū
5batch_instance_normalization_508/moments/StopGradientStopGradient6batch_instance_normalization_508/moments/mean:output:0*
T0*&
_output_shapes
:@į
:batch_instance_normalization_508/moments/SquaredDifferenceSquaredDifferenceconv2d_616/Conv2D:output:0>batch_instance_normalization_508/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Cbatch_instance_normalization_508/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_508/moments/varianceMean>batch_instance_normalization_508/moments/SquaredDifference:z:0Lbatch_instance_normalization_508/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(ŧ
$batch_instance_normalization_508/subSubconv2d_616/Conv2D:output:06batch_instance_normalization_508/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@k
&batch_instance_normalization_508/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ë
$batch_instance_normalization_508/addAddV2:batch_instance_normalization_508/moments/variance:output:0/batch_instance_normalization_508/add/y:output:0*
T0*&
_output_shapes
:@
&batch_instance_normalization_508/RsqrtRsqrt(batch_instance_normalization_508/add:z:0*
T0*&
_output_shapes
:@―
$batch_instance_normalization_508/mulMul(batch_instance_normalization_508/sub:z:0*batch_instance_normalization_508/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Abatch_instance_normalization_508/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ę
/batch_instance_normalization_508/moments_1/meanMeanconv2d_616/Conv2D:output:0Jbatch_instance_normalization_508/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(ŧ
7batch_instance_normalization_508/moments_1/StopGradientStopGradient8batch_instance_normalization_508/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ë
<batch_instance_normalization_508/moments_1/SquaredDifferenceSquaredDifferenceconv2d_616/Conv2D:output:0@batch_instance_normalization_508/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Ebatch_instance_normalization_508/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_508/moments_1/varianceMean@batch_instance_normalization_508/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_508/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(ŋ
&batch_instance_normalization_508/sub_1Subconv2d_616/Conv2D:output:08batch_instance_normalization_508/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@m
(batch_instance_normalization_508/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ú
&batch_instance_normalization_508/add_1AddV2<batch_instance_normalization_508/moments_1/variance:output:01batch_instance_normalization_508/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
(batch_instance_normalization_508/Rsqrt_1Rsqrt*batch_instance_normalization_508/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ã
&batch_instance_normalization_508/mul_1Mul*batch_instance_normalization_508/sub_1:z:0,batch_instance_normalization_508/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ī
/batch_instance_normalization_508/ReadVariableOpReadVariableOp8batch_instance_normalization_508_readvariableop_resource*
_output_shapes
:@*
dtype0Ė
&batch_instance_normalization_508/mul_2Mul7batch_instance_normalization_508/ReadVariableOp:value:0(batch_instance_normalization_508/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ķ
1batch_instance_normalization_508/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_508_readvariableop_resource*
_output_shapes
:@*
dtype0m
(batch_instance_normalization_508/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ā
&batch_instance_normalization_508/sub_2Sub1batch_instance_normalization_508/sub_2/x:output:09batch_instance_normalization_508/ReadVariableOp_1:value:0*
T0*
_output_shapes
:@Á
&batch_instance_normalization_508/mul_3Mul*batch_instance_normalization_508/sub_2:z:0*batch_instance_normalization_508/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ã
&batch_instance_normalization_508/add_2AddV2*batch_instance_normalization_508/mul_2:z:0*batch_instance_normalization_508/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@°
5batch_instance_normalization_508/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_508_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0Ô
&batch_instance_normalization_508/mul_4Mul*batch_instance_normalization_508/add_2:z:0=batch_instance_normalization_508/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@°
5batch_instance_normalization_508/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_508_add_3_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
&batch_instance_normalization_508/add_3AddV2*batch_instance_normalization_508/mul_4:z:0=batch_instance_normalization_508/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@w
LeakyRelu_9	LeakyRelu*batch_instance_normalization_508/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 conv2d_617/Conv2D/ReadVariableOpReadVariableOp)conv2d_617_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ä
conv2d_617/Conv2DConv2DLeakyRelu_9:activations:0(conv2d_617/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides

?batch_instance_normalization_509/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ý
-batch_instance_normalization_509/moments/meanMeanconv2d_617/Conv2D:output:0Hbatch_instance_normalization_509/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(Ū
5batch_instance_normalization_509/moments/StopGradientStopGradient6batch_instance_normalization_509/moments/mean:output:0*
T0*&
_output_shapes
:@į
:batch_instance_normalization_509/moments/SquaredDifferenceSquaredDifferenceconv2d_617/Conv2D:output:0>batch_instance_normalization_509/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Cbatch_instance_normalization_509/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_509/moments/varianceMean>batch_instance_normalization_509/moments/SquaredDifference:z:0Lbatch_instance_normalization_509/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(ŧ
$batch_instance_normalization_509/subSubconv2d_617/Conv2D:output:06batch_instance_normalization_509/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@k
&batch_instance_normalization_509/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ë
$batch_instance_normalization_509/addAddV2:batch_instance_normalization_509/moments/variance:output:0/batch_instance_normalization_509/add/y:output:0*
T0*&
_output_shapes
:@
&batch_instance_normalization_509/RsqrtRsqrt(batch_instance_normalization_509/add:z:0*
T0*&
_output_shapes
:@―
$batch_instance_normalization_509/mulMul(batch_instance_normalization_509/sub:z:0*batch_instance_normalization_509/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Abatch_instance_normalization_509/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ę
/batch_instance_normalization_509/moments_1/meanMeanconv2d_617/Conv2D:output:0Jbatch_instance_normalization_509/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(ŧ
7batch_instance_normalization_509/moments_1/StopGradientStopGradient8batch_instance_normalization_509/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@ë
<batch_instance_normalization_509/moments_1/SquaredDifferenceSquaredDifferenceconv2d_617/Conv2D:output:0@batch_instance_normalization_509/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
Ebatch_instance_normalization_509/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_509/moments_1/varianceMean@batch_instance_normalization_509/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_509/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(ŋ
&batch_instance_normalization_509/sub_1Subconv2d_617/Conv2D:output:08batch_instance_normalization_509/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@m
(batch_instance_normalization_509/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ú
&batch_instance_normalization_509/add_1AddV2<batch_instance_normalization_509/moments_1/variance:output:01batch_instance_normalization_509/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
(batch_instance_normalization_509/Rsqrt_1Rsqrt*batch_instance_normalization_509/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ã
&batch_instance_normalization_509/mul_1Mul*batch_instance_normalization_509/sub_1:z:0,batch_instance_normalization_509/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ī
/batch_instance_normalization_509/ReadVariableOpReadVariableOp8batch_instance_normalization_509_readvariableop_resource*
_output_shapes
:@*
dtype0Ė
&batch_instance_normalization_509/mul_2Mul7batch_instance_normalization_509/ReadVariableOp:value:0(batch_instance_normalization_509/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ķ
1batch_instance_normalization_509/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_509_readvariableop_resource*
_output_shapes
:@*
dtype0m
(batch_instance_normalization_509/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ā
&batch_instance_normalization_509/sub_2Sub1batch_instance_normalization_509/sub_2/x:output:09batch_instance_normalization_509/ReadVariableOp_1:value:0*
T0*
_output_shapes
:@Á
&batch_instance_normalization_509/mul_3Mul*batch_instance_normalization_509/sub_2:z:0*batch_instance_normalization_509/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@Ã
&batch_instance_normalization_509/add_2AddV2*batch_instance_normalization_509/mul_2:z:0*batch_instance_normalization_509/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@°
5batch_instance_normalization_509/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_509_mul_4_readvariableop_resource*
_output_shapes
:@*
dtype0Ô
&batch_instance_normalization_509/mul_4Mul*batch_instance_normalization_509/add_2:z:0=batch_instance_normalization_509/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@°
5batch_instance_normalization_509/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_509_add_3_readvariableop_resource*
_output_shapes
:@*
dtype0Ö
&batch_instance_normalization_509/add_3AddV2*batch_instance_normalization_509/mul_4:z:0=batch_instance_normalization_509/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@x
LeakyRelu_10	LeakyRelu*batch_instance_normalization_509/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 conv2d_618/Conv2D/ReadVariableOpReadVariableOp)conv2d_618_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Å
conv2d_618/Conv2DConv2DLeakyRelu_10:activations:0(conv2d_618/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides

?batch_instance_normalization_510/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ý
-batch_instance_normalization_510/moments/meanMeanconv2d_618/Conv2D:output:0Hbatch_instance_normalization_510/moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(Ū
5batch_instance_normalization_510/moments/StopGradientStopGradient6batch_instance_normalization_510/moments/mean:output:0*
T0*&
_output_shapes
:į
:batch_instance_normalization_510/moments/SquaredDifferenceSquaredDifferenceconv2d_618/Conv2D:output:0>batch_instance_normalization_510/moments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Cbatch_instance_normalization_510/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
1batch_instance_normalization_510/moments/varianceMean>batch_instance_normalization_510/moments/SquaredDifference:z:0Lbatch_instance_normalization_510/moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(ŧ
$batch_instance_normalization_510/subSubconv2d_618/Conv2D:output:06batch_instance_normalization_510/moments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸk
&batch_instance_normalization_510/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ë
$batch_instance_normalization_510/addAddV2:batch_instance_normalization_510/moments/variance:output:0/batch_instance_normalization_510/add/y:output:0*
T0*&
_output_shapes
:
&batch_instance_normalization_510/RsqrtRsqrt(batch_instance_normalization_510/add:z:0*
T0*&
_output_shapes
:―
$batch_instance_normalization_510/mulMul(batch_instance_normalization_510/sub:z:0*batch_instance_normalization_510/Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Abatch_instance_normalization_510/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ę
/batch_instance_normalization_510/moments_1/meanMeanconv2d_618/Conv2D:output:0Jbatch_instance_normalization_510/moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ŧ
7batch_instance_normalization_510/moments_1/StopGradientStopGradient8batch_instance_normalization_510/moments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸë
<batch_instance_normalization_510/moments_1/SquaredDifferenceSquaredDifferenceconv2d_618/Conv2D:output:0@batch_instance_normalization_510/moments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
Ebatch_instance_normalization_510/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
3batch_instance_normalization_510/moments_1/varianceMean@batch_instance_normalization_510/moments_1/SquaredDifference:z:0Nbatch_instance_normalization_510/moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
	keep_dims(ŋ
&batch_instance_normalization_510/sub_1Subconv2d_618/Conv2D:output:08batch_instance_normalization_510/moments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸm
(batch_instance_normalization_510/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7Ú
&batch_instance_normalization_510/add_1AddV2<batch_instance_normalization_510/moments_1/variance:output:01batch_instance_normalization_510/add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
(batch_instance_normalization_510/Rsqrt_1Rsqrt*batch_instance_normalization_510/add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸÃ
&batch_instance_normalization_510/mul_1Mul*batch_instance_normalization_510/sub_1:z:0,batch_instance_normalization_510/Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸĪ
/batch_instance_normalization_510/ReadVariableOpReadVariableOp8batch_instance_normalization_510_readvariableop_resource*
_output_shapes
:*
dtype0Ė
&batch_instance_normalization_510/mul_2Mul7batch_instance_normalization_510/ReadVariableOp:value:0(batch_instance_normalization_510/mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸĶ
1batch_instance_normalization_510/ReadVariableOp_1ReadVariableOp8batch_instance_normalization_510_readvariableop_resource*
_output_shapes
:*
dtype0m
(batch_instance_normalization_510/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ā
&batch_instance_normalization_510/sub_2Sub1batch_instance_normalization_510/sub_2/x:output:09batch_instance_normalization_510/ReadVariableOp_1:value:0*
T0*
_output_shapes
:Á
&batch_instance_normalization_510/mul_3Mul*batch_instance_normalization_510/sub_2:z:0*batch_instance_normalization_510/mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸÃ
&batch_instance_normalization_510/add_2AddV2*batch_instance_normalization_510/mul_2:z:0*batch_instance_normalization_510/mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ°
5batch_instance_normalization_510/mul_4/ReadVariableOpReadVariableOp>batch_instance_normalization_510_mul_4_readvariableop_resource*
_output_shapes
:*
dtype0Ô
&batch_instance_normalization_510/mul_4Mul*batch_instance_normalization_510/add_2:z:0=batch_instance_normalization_510/mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ°
5batch_instance_normalization_510/add_3/ReadVariableOpReadVariableOp>batch_instance_normalization_510_add_3_readvariableop_resource*
_output_shapes
:*
dtype0Ö
&batch_instance_normalization_510/add_3AddV2*batch_instance_normalization_510/mul_4:z:0=batch_instance_normalization_510/add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸx
LeakyRelu_11	LeakyRelu*batch_instance_normalization_510/add_3:z:0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 conv2d_619/Conv2D/ReadVariableOpReadVariableOp)conv2d_619_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Å
conv2d_619/Conv2DConv2DLeakyRelu_11:activations:0(conv2d_619/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*
paddingSAME*
strides
d
TanhTanhconv2d_619/Conv2D:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸa
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸÐ
NoOpNoOp0^batch_instance_normalization_500/ReadVariableOp2^batch_instance_normalization_500/ReadVariableOp_16^batch_instance_normalization_500/add_3/ReadVariableOp6^batch_instance_normalization_500/mul_4/ReadVariableOp0^batch_instance_normalization_501/ReadVariableOp2^batch_instance_normalization_501/ReadVariableOp_16^batch_instance_normalization_501/add_3/ReadVariableOp6^batch_instance_normalization_501/mul_4/ReadVariableOp0^batch_instance_normalization_502/ReadVariableOp2^batch_instance_normalization_502/ReadVariableOp_16^batch_instance_normalization_502/add_3/ReadVariableOp6^batch_instance_normalization_502/mul_4/ReadVariableOp0^batch_instance_normalization_503/ReadVariableOp2^batch_instance_normalization_503/ReadVariableOp_16^batch_instance_normalization_503/add_3/ReadVariableOp6^batch_instance_normalization_503/mul_4/ReadVariableOp0^batch_instance_normalization_504/ReadVariableOp2^batch_instance_normalization_504/ReadVariableOp_16^batch_instance_normalization_504/add_3/ReadVariableOp6^batch_instance_normalization_504/mul_4/ReadVariableOp0^batch_instance_normalization_505/ReadVariableOp2^batch_instance_normalization_505/ReadVariableOp_16^batch_instance_normalization_505/add_3/ReadVariableOp6^batch_instance_normalization_505/mul_4/ReadVariableOp0^batch_instance_normalization_506/ReadVariableOp2^batch_instance_normalization_506/ReadVariableOp_16^batch_instance_normalization_506/add_3/ReadVariableOp6^batch_instance_normalization_506/mul_4/ReadVariableOp0^batch_instance_normalization_507/ReadVariableOp2^batch_instance_normalization_507/ReadVariableOp_16^batch_instance_normalization_507/add_3/ReadVariableOp6^batch_instance_normalization_507/mul_4/ReadVariableOp0^batch_instance_normalization_508/ReadVariableOp2^batch_instance_normalization_508/ReadVariableOp_16^batch_instance_normalization_508/add_3/ReadVariableOp6^batch_instance_normalization_508/mul_4/ReadVariableOp0^batch_instance_normalization_509/ReadVariableOp2^batch_instance_normalization_509/ReadVariableOp_16^batch_instance_normalization_509/add_3/ReadVariableOp6^batch_instance_normalization_509/mul_4/ReadVariableOp0^batch_instance_normalization_510/ReadVariableOp2^batch_instance_normalization_510/ReadVariableOp_16^batch_instance_normalization_510/add_3/ReadVariableOp6^batch_instance_normalization_510/mul_4/ReadVariableOp!^conv2d_606/Conv2D/ReadVariableOp!^conv2d_607/Conv2D/ReadVariableOp!^conv2d_608/Conv2D/ReadVariableOp!^conv2d_609/Conv2D/ReadVariableOp!^conv2d_610/Conv2D/ReadVariableOp!^conv2d_611/Conv2D/ReadVariableOp!^conv2d_612/Conv2D/ReadVariableOp!^conv2d_613/Conv2D/ReadVariableOp!^conv2d_614/Conv2D/ReadVariableOp!^conv2d_615/Conv2D/ReadVariableOp!^conv2d_616/Conv2D/ReadVariableOp!^conv2d_617/Conv2D/ReadVariableOp!^conv2d_618/Conv2D/ReadVariableOp!^conv2d_619/Conv2D/ReadVariableOp5^conv2d_transpose_100/conv2d_transpose/ReadVariableOp5^conv2d_transpose_101/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*ą
_input_shapes
:ĸĸĸĸĸĸĸĸĸ:ĸĸĸĸĸĸĸĸĸ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2b
/batch_instance_normalization_500/ReadVariableOp/batch_instance_normalization_500/ReadVariableOp2f
1batch_instance_normalization_500/ReadVariableOp_11batch_instance_normalization_500/ReadVariableOp_12n
5batch_instance_normalization_500/add_3/ReadVariableOp5batch_instance_normalization_500/add_3/ReadVariableOp2n
5batch_instance_normalization_500/mul_4/ReadVariableOp5batch_instance_normalization_500/mul_4/ReadVariableOp2b
/batch_instance_normalization_501/ReadVariableOp/batch_instance_normalization_501/ReadVariableOp2f
1batch_instance_normalization_501/ReadVariableOp_11batch_instance_normalization_501/ReadVariableOp_12n
5batch_instance_normalization_501/add_3/ReadVariableOp5batch_instance_normalization_501/add_3/ReadVariableOp2n
5batch_instance_normalization_501/mul_4/ReadVariableOp5batch_instance_normalization_501/mul_4/ReadVariableOp2b
/batch_instance_normalization_502/ReadVariableOp/batch_instance_normalization_502/ReadVariableOp2f
1batch_instance_normalization_502/ReadVariableOp_11batch_instance_normalization_502/ReadVariableOp_12n
5batch_instance_normalization_502/add_3/ReadVariableOp5batch_instance_normalization_502/add_3/ReadVariableOp2n
5batch_instance_normalization_502/mul_4/ReadVariableOp5batch_instance_normalization_502/mul_4/ReadVariableOp2b
/batch_instance_normalization_503/ReadVariableOp/batch_instance_normalization_503/ReadVariableOp2f
1batch_instance_normalization_503/ReadVariableOp_11batch_instance_normalization_503/ReadVariableOp_12n
5batch_instance_normalization_503/add_3/ReadVariableOp5batch_instance_normalization_503/add_3/ReadVariableOp2n
5batch_instance_normalization_503/mul_4/ReadVariableOp5batch_instance_normalization_503/mul_4/ReadVariableOp2b
/batch_instance_normalization_504/ReadVariableOp/batch_instance_normalization_504/ReadVariableOp2f
1batch_instance_normalization_504/ReadVariableOp_11batch_instance_normalization_504/ReadVariableOp_12n
5batch_instance_normalization_504/add_3/ReadVariableOp5batch_instance_normalization_504/add_3/ReadVariableOp2n
5batch_instance_normalization_504/mul_4/ReadVariableOp5batch_instance_normalization_504/mul_4/ReadVariableOp2b
/batch_instance_normalization_505/ReadVariableOp/batch_instance_normalization_505/ReadVariableOp2f
1batch_instance_normalization_505/ReadVariableOp_11batch_instance_normalization_505/ReadVariableOp_12n
5batch_instance_normalization_505/add_3/ReadVariableOp5batch_instance_normalization_505/add_3/ReadVariableOp2n
5batch_instance_normalization_505/mul_4/ReadVariableOp5batch_instance_normalization_505/mul_4/ReadVariableOp2b
/batch_instance_normalization_506/ReadVariableOp/batch_instance_normalization_506/ReadVariableOp2f
1batch_instance_normalization_506/ReadVariableOp_11batch_instance_normalization_506/ReadVariableOp_12n
5batch_instance_normalization_506/add_3/ReadVariableOp5batch_instance_normalization_506/add_3/ReadVariableOp2n
5batch_instance_normalization_506/mul_4/ReadVariableOp5batch_instance_normalization_506/mul_4/ReadVariableOp2b
/batch_instance_normalization_507/ReadVariableOp/batch_instance_normalization_507/ReadVariableOp2f
1batch_instance_normalization_507/ReadVariableOp_11batch_instance_normalization_507/ReadVariableOp_12n
5batch_instance_normalization_507/add_3/ReadVariableOp5batch_instance_normalization_507/add_3/ReadVariableOp2n
5batch_instance_normalization_507/mul_4/ReadVariableOp5batch_instance_normalization_507/mul_4/ReadVariableOp2b
/batch_instance_normalization_508/ReadVariableOp/batch_instance_normalization_508/ReadVariableOp2f
1batch_instance_normalization_508/ReadVariableOp_11batch_instance_normalization_508/ReadVariableOp_12n
5batch_instance_normalization_508/add_3/ReadVariableOp5batch_instance_normalization_508/add_3/ReadVariableOp2n
5batch_instance_normalization_508/mul_4/ReadVariableOp5batch_instance_normalization_508/mul_4/ReadVariableOp2b
/batch_instance_normalization_509/ReadVariableOp/batch_instance_normalization_509/ReadVariableOp2f
1batch_instance_normalization_509/ReadVariableOp_11batch_instance_normalization_509/ReadVariableOp_12n
5batch_instance_normalization_509/add_3/ReadVariableOp5batch_instance_normalization_509/add_3/ReadVariableOp2n
5batch_instance_normalization_509/mul_4/ReadVariableOp5batch_instance_normalization_509/mul_4/ReadVariableOp2b
/batch_instance_normalization_510/ReadVariableOp/batch_instance_normalization_510/ReadVariableOp2f
1batch_instance_normalization_510/ReadVariableOp_11batch_instance_normalization_510/ReadVariableOp_12n
5batch_instance_normalization_510/add_3/ReadVariableOp5batch_instance_normalization_510/add_3/ReadVariableOp2n
5batch_instance_normalization_510/mul_4/ReadVariableOp5batch_instance_normalization_510/mul_4/ReadVariableOp2D
 conv2d_606/Conv2D/ReadVariableOp conv2d_606/Conv2D/ReadVariableOp2D
 conv2d_607/Conv2D/ReadVariableOp conv2d_607/Conv2D/ReadVariableOp2D
 conv2d_608/Conv2D/ReadVariableOp conv2d_608/Conv2D/ReadVariableOp2D
 conv2d_609/Conv2D/ReadVariableOp conv2d_609/Conv2D/ReadVariableOp2D
 conv2d_610/Conv2D/ReadVariableOp conv2d_610/Conv2D/ReadVariableOp2D
 conv2d_611/Conv2D/ReadVariableOp conv2d_611/Conv2D/ReadVariableOp2D
 conv2d_612/Conv2D/ReadVariableOp conv2d_612/Conv2D/ReadVariableOp2D
 conv2d_613/Conv2D/ReadVariableOp conv2d_613/Conv2D/ReadVariableOp2D
 conv2d_614/Conv2D/ReadVariableOp conv2d_614/Conv2D/ReadVariableOp2D
 conv2d_615/Conv2D/ReadVariableOp conv2d_615/Conv2D/ReadVariableOp2D
 conv2d_616/Conv2D/ReadVariableOp conv2d_616/Conv2D/ReadVariableOp2D
 conv2d_617/Conv2D/ReadVariableOp conv2d_617/Conv2D/ReadVariableOp2D
 conv2d_618/Conv2D/ReadVariableOp conv2d_618/Conv2D/ReadVariableOp2D
 conv2d_619/Conv2D/ReadVariableOp conv2d_619/Conv2D/ReadVariableOp2l
4conv2d_transpose_100/conv2d_transpose/ReadVariableOp4conv2d_transpose_100/conv2d_transpose/ReadVariableOp2l
4conv2d_transpose_101/conv2d_transpose/ReadVariableOp4conv2d_transpose_101/conv2d_transpose/ReadVariableOp:[ W
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
"
_user_specified_name
inputs/1
ģ$
Ë
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56545894
x%
readvariableop_resource:@+
mul_4_readvariableop_resource:@+
add_3_readvariableop_resource:@
identityĒReadVariableOpĒReadVariableOp_1Ēadd_3/ReadVariableOpĒmul_4/ReadVariableOps
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          
moments/meanMeanx'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(l
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferencexmoments/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@w
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          Ķ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:@*
	keep_dims(`
subSubxmoments/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7h
addAddV2moments/variance:output:0add/y:output:0*
T0*&
_output_shapes
:@H
RsqrtRsqrtadd:z:0*
T0*&
_output_shapes
:@Z
mulMulsub:z:0	Rsqrt:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@q
 moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      
moments_1/meanMeanx)moments_1/mean/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(y
moments_1/StopGradientStopGradientmoments_1/mean:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
moments_1/SquaredDifferenceSquaredDifferencexmoments_1/StopGradient:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@u
$moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ĩ
moments_1/varianceMeanmoments_1/SquaredDifference:z:0-moments_1/variance/reduction_indices:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
	keep_dims(d
sub_1Subxmoments_1/mean:output:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŽÅ'7w
add_1AddV2moments_1/variance:output:0add_1/y:output:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@U
Rsqrt_1Rsqrt	add_1:z:0*
T0*/
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
mul_1Mul	sub_1:z:0Rsqrt_1:y:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0i
mul_2MulReadVariableOp:value:0mul:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0L
sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
sub_2Subsub_2/x:output:0ReadVariableOp_1:value:0*
T0*
_output_shapes
:@^
mul_3Mul	sub_2:z:0	mul_1:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@`
add_2AddV2	mul_2:z:0	mul_3:z:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@n
mul_4/ReadVariableOpReadVariableOpmul_4_readvariableop_resource*
_output_shapes
:@*
dtype0q
mul_4Mul	add_2:z:0mul_4/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@n
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*
_output_shapes
:@*
dtype0s
add_3AddV2	mul_4:z:0add_3/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@b
IdentityIdentity	add_3:z:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^add_3/ReadVariableOp^mul_4/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ĸĸĸĸĸĸĸĸĸ@: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
mul_4/ReadVariableOpmul_4/ReadVariableOp:T P
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@

_user_specified_namex
Õ

-__inference_conv2d_619_layer_call_fn_56545966

inputs!
unknown:
identityĒStatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56542556y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Š
đ
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56542436

inputs8
conv2d_readvariableop_resource:@@
identityĒConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@*
paddingSAME*
strides
h
IdentityIdentityConv2D:output:0^NoOp*
T0*1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ĸĸĸĸĸĸĸĸĸ@: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@
 
_user_specified_nameinputs
Ã
P
4__inference_max_pooling2d_101_layer_call_fn_56545247

inputs
identityā
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56541714
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ:r n
J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
Ã
P
4__inference_max_pooling2d_100_layer_call_fn_56545107

inputs
identityā
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *X
fSRQ
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56541702
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ:r n
J
_output_shapes8
6:4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 
_user_specified_nameinputs
ĩ
Ã
C__inference_batch_instance_normalization_507_layer_call_fn_56545687
x
unknown:	
	unknown_0:	
	unknown_1:	
identityĒStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *g
fbR`
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56542355x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":ĸĸĸĸĸĸĸĸĸ@@: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
0
_output_shapes
:ĸĸĸĸĸĸĸĸĸ@@

_user_specified_namex"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_defaultō
E
input_1:
serving_default_input_1:0ĸĸĸĸĸĸĸĸĸ
E
input_2:
serving_default_input_2:0ĸĸĸĸĸĸĸĸĸF
output_1:
StatefulPartitionedCall:0ĸĸĸĸĸĸĸĸĸtensorflow/serving/predict:Îž
Ŧ
conv1_1
conv1_2
	pool1
conv2_1
	bn2_1
conv2_2
	bn2_2
	pool2
	conv3_1
	
bn3_1
conv3_2
	bn3_2
conv3_3
	bn3_3
conv3_4
	bn3_4

convt1
conv6_1
	bn6_1
conv6_2
	bn6_2

convt2
conv7_1
	bn7_1
conv7_2
	bn7_2
conv7_3
	bn7_3
out
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%
signatures"
_tf_keras_model
ą

&kernel
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
ą

-kernel
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses"
_tf_keras_layer
Ĩ
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses"
_tf_keras_layer
ą

:kernel
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Arho
	Bgamma
Cbeta
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
ą

Jkernel
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
Qrho
	Rgamma
Sbeta
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_layer
Ĩ
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses"
_tf_keras_layer
ą

`kernel
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
grho
	hgamma
ibeta
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
ą

pkernel
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer
Ã
wrho
	xgamma
ybeta
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ė
rho

gamma
	beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ė
rho

gamma
	beta
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
 kernel
Ą	variables
Ētrainable_variables
Ģregularization_losses
Ī	keras_api
Ĩ__call__
+Ķ&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
§kernel
Ļ	variables
Đtrainable_variables
Šregularization_losses
Ŧ	keras_api
Ž__call__
+­&call_and_return_all_conditional_losses"
_tf_keras_layer
Ė
Ūrho

Ŋgamma
	°beta
ą	variables
ētrainable_variables
ģregularization_losses
ī	keras_api
ĩ__call__
+ķ&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
·kernel
ļ	variables
đtrainable_variables
šregularization_losses
ŧ	keras_api
ž__call__
+―&call_and_return_all_conditional_losses"
_tf_keras_layer
Ė
ūrho

ŋgamma
	Ābeta
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
Įkernel
Č	variables
Étrainable_variables
Ęregularization_losses
Ë	keras_api
Ė__call__
+Í&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
Îkernel
Ï	variables
Ðtrainable_variables
Ņregularization_losses
Ō	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_tf_keras_layer
Ė
Õrho

Ögamma
	Ũbeta
Ø	variables
Ųtrainable_variables
Úregularization_losses
Û	keras_api
Ü__call__
+Ý&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
Þkernel
ß	variables
ātrainable_variables
áregularization_losses
â	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"
_tf_keras_layer
Ė
årho

ægamma
	įbeta
č	variables
étrainable_variables
ęregularization_losses
ë	keras_api
ė__call__
+í&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
îkernel
ï	variables
ðtrainable_variables
ņregularization_losses
ō	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"
_tf_keras_layer
Ė
õrho

ögamma
	ũbeta
ø	variables
ųtrainable_variables
úregularization_losses
û	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"
_tf_keras_layer
ļ
þkernel
ĸ	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
―
&0
-1
:2
A3
B4
C5
J6
Q7
R8
S9
`10
g11
h12
i13
p14
w15
x16
y17
18
19
20
21
22
23
24
25
 26
§27
Ū28
Ŋ29
°30
·31
ū32
ŋ33
Ā34
Į35
Î36
Õ37
Ö38
Ũ39
Þ40
å41
æ42
į43
î44
õ45
ö46
ũ47
þ48"
trackable_list_wrapper
―
&0
-1
:2
A3
B4
C5
J6
Q7
R8
S9
`10
g11
h12
i13
p14
w15
x16
y17
18
19
20
21
22
23
24
25
 26
§27
Ū28
Ŋ29
°30
·31
ū32
ŋ33
Ā34
Į35
Î36
Õ37
Ö38
Ũ39
Þ40
å41
æ42
į43
î44
õ45
ö46
ũ47
þ48"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ō2ï
,__inference_face_g_18_layer_call_fn_56542663
,__inference_face_g_18_layer_call_fn_56543776
,__inference_face_g_18_layer_call_fn_56543880
,__inference_face_g_18_layer_call_fn_56543370ī
Ŧē§
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
kwonlydefaultsŠ 
annotationsŠ *
 
Þ2Û
G__inference_face_g_18_layer_call_and_return_conditional_losses_56544424
G__inference_face_g_18_layer_call_and_return_conditional_losses_56544968
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543521
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543672ī
Ŧē§
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
kwonlydefaultsŠ 
annotationsŠ *
 
ŨBÔ
#__inference__wrapped_model_56541693input_1input_2"
ē
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŠ *
 
-
serving_default"
signature_map
+:)@2conv2d_606/kernel
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_606_layer_call_fn_56545081Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56545088Ē
ē
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
annotationsŠ *
 
+:)@@2conv2d_607/kernel
'
-0"
trackable_list_wrapper
'
-0"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_607_layer_call_fn_56545095Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56545102Ē
ē
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
annotationsŠ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ē
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
Þ2Û
4__inference_max_pooling2d_100_layer_call_fn_56545107Ē
ē
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
annotationsŠ *
 
ų2ö
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56545112Ē
ē
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
annotationsŠ *
 
,:*@2conv2d_608/kernel
'
:0"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_608_layer_call_fn_56545119Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56545126Ē
ē
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
annotationsŠ *
 
3:12$batch_instance_normalization_500/rho
5:32&batch_instance_normalization_500/gamma
4:22%batch_instance_normalization_500/beta
5
A0
B1
C2"
trackable_list_wrapper
5
A0
B1
C2"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
non_trainable_variables
 layers
Ąmetrics
 Ēlayer_regularization_losses
Ģlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_500_layer_call_fn_56545137
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56545177
ē
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
annotationsŠ *
 
-:+2conv2d_609/kernel
'
J0"
trackable_list_wrapper
'
J0"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
Īnon_trainable_variables
Ĩlayers
Ķmetrics
 §layer_regularization_losses
Ļlayer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_609_layer_call_fn_56545184Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56545191Ē
ē
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
annotationsŠ *
 
3:12$batch_instance_normalization_501/rho
5:32&batch_instance_normalization_501/gamma
4:22%batch_instance_normalization_501/beta
5
Q0
R1
S2"
trackable_list_wrapper
5
Q0
R1
S2"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
Đnon_trainable_variables
Šlayers
Ŧmetrics
 Žlayer_regularization_losses
­layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_501_layer_call_fn_56545202
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56545242
ē
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
annotationsŠ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ē
Ūnon_trainable_variables
Ŋlayers
°metrics
 ąlayer_regularization_losses
ēlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
Þ2Û
4__inference_max_pooling2d_101_layer_call_fn_56545247Ē
ē
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
annotationsŠ *
 
ų2ö
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56545252Ē
ē
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
annotationsŠ *
 
-:+2conv2d_610/kernel
'
`0"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
ģnon_trainable_variables
īlayers
ĩmetrics
 ķlayer_regularization_losses
·layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_610_layer_call_fn_56545259Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56545278Ē
ē
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
annotationsŠ *
 
3:12$batch_instance_normalization_502/rho
5:32&batch_instance_normalization_502/gamma
4:22%batch_instance_normalization_502/beta
5
g0
h1
i2"
trackable_list_wrapper
5
g0
h1
i2"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
ļnon_trainable_variables
đlayers
šmetrics
 ŧlayer_regularization_losses
žlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_502_layer_call_fn_56545289
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56545329
ē
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
annotationsŠ *
 
-:+2conv2d_611/kernel
'
p0"
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
―non_trainable_variables
ūlayers
ŋmetrics
 Ālayer_regularization_losses
Álayer_metrics
q	variables
rtrainable_variables
sregularization_losses
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_611_layer_call_fn_56545336Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56545355Ē
ē
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
annotationsŠ *
 
3:12$batch_instance_normalization_503/rho
5:32&batch_instance_normalization_503/gamma
4:22%batch_instance_normalization_503/beta
5
w0
x1
y2"
trackable_list_wrapper
5
w0
x1
y2"
trackable_list_wrapper
 "
trackable_list_wrapper
ē
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_503_layer_call_fn_56545366
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56545406
ē
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
annotationsŠ *
 
-:+2conv2d_612/kernel
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
Įnon_trainable_variables
Člayers
Émetrics
 Ęlayer_regularization_losses
Ëlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_612_layer_call_fn_56545413Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56545432Ē
ē
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
annotationsŠ *
 
3:12$batch_instance_normalization_504/rho
5:32&batch_instance_normalization_504/gamma
4:22%batch_instance_normalization_504/beta
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
Ėnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_504_layer_call_fn_56545443
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56545483
ē
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
annotationsŠ *
 
-:+2conv2d_613/kernel
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
Ņnon_trainable_variables
Ōlayers
Ómetrics
 Ôlayer_regularization_losses
Õlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_613_layer_call_fn_56545490Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56545509Ē
ē
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
annotationsŠ *
 
3:12$batch_instance_normalization_505/rho
5:32&batch_instance_normalization_505/gamma
4:22%batch_instance_normalization_505/beta
8
0
1
2"
trackable_list_wrapper
8
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
Önon_trainable_variables
Ũlayers
Ømetrics
 Ųlayer_regularization_losses
Úlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_505_layer_call_fn_56545520
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56545560
ē
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
annotationsŠ *
 
7:52conv2d_transpose_100/kernel
(
 0"
trackable_list_wrapper
(
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
Ą	variables
Ētrainable_variables
Ģregularization_losses
Ĩ__call__
+Ķ&call_and_return_all_conditional_losses
'Ķ"call_and_return_conditional_losses"
_generic_user_object
á2Þ
7__inference_conv2d_transpose_100_layer_call_fn_56545567Ē
ē
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
annotationsŠ *
 
ü2ų
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56545597Ē
ē
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
annotationsŠ *
 
-:+2conv2d_614/kernel
(
§0"
trackable_list_wrapper
(
§0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
ānon_trainable_variables
álayers
âmetrics
 ãlayer_regularization_losses
älayer_metrics
Ļ	variables
Đtrainable_variables
Šregularization_losses
Ž__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_614_layer_call_fn_56545604Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56545611Ē
ē
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
annotationsŠ *
 
3:12$batch_instance_normalization_506/rho
5:32&batch_instance_normalization_506/gamma
4:22%batch_instance_normalization_506/beta
8
Ū0
Ŋ1
°2"
trackable_list_wrapper
8
Ū0
Ŋ1
°2"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
ånon_trainable_variables
ælayers
įmetrics
 člayer_regularization_losses
élayer_metrics
ą	variables
ētrainable_variables
ģregularization_losses
ĩ__call__
+ķ&call_and_return_all_conditional_losses
'ķ"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_506_layer_call_fn_56545622
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56545662
ē
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
annotationsŠ *
 
-:+2conv2d_615/kernel
(
·0"
trackable_list_wrapper
(
·0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
ęnon_trainable_variables
ëlayers
ėmetrics
 ílayer_regularization_losses
îlayer_metrics
ļ	variables
đtrainable_variables
šregularization_losses
ž__call__
+―&call_and_return_all_conditional_losses
'―"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_615_layer_call_fn_56545669Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56545676Ē
ē
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
annotationsŠ *
 
3:12$batch_instance_normalization_507/rho
5:32&batch_instance_normalization_507/gamma
4:22%batch_instance_normalization_507/beta
8
ū0
ŋ1
Ā2"
trackable_list_wrapper
8
ū0
ŋ1
Ā2"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
ïnon_trainable_variables
ðlayers
ņmetrics
 ōlayer_regularization_losses
ólayer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_507_layer_call_fn_56545687
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56545727
ē
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
annotationsŠ *
 
6:4@2conv2d_transpose_101/kernel
(
Į0"
trackable_list_wrapper
(
Į0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
ônon_trainable_variables
õlayers
ömetrics
 ũlayer_regularization_losses
ølayer_metrics
Č	variables
Étrainable_variables
Ęregularization_losses
Ė__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
á2Þ
7__inference_conv2d_transpose_101_layer_call_fn_56545734Ē
ē
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
annotationsŠ *
 
ü2ų
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56545764Ē
ē
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
annotationsŠ *
 
,:*@2conv2d_616/kernel
(
Î0"
trackable_list_wrapper
(
Î0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
ųnon_trainable_variables
úlayers
ûmetrics
 ülayer_regularization_losses
ýlayer_metrics
Ï	variables
Ðtrainable_variables
Ņregularization_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses
'Ô"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_616_layer_call_fn_56545771Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56545778Ē
ē
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
annotationsŠ *
 
2:0@2$batch_instance_normalization_508/rho
4:2@2&batch_instance_normalization_508/gamma
3:1@2%batch_instance_normalization_508/beta
8
Õ0
Ö1
Ũ2"
trackable_list_wrapper
8
Õ0
Ö1
Ũ2"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
þnon_trainable_variables
ĸlayers
metrics
 layer_regularization_losses
layer_metrics
Ø	variables
Ųtrainable_variables
Úregularization_losses
Ü__call__
+Ý&call_and_return_all_conditional_losses
'Ý"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_508_layer_call_fn_56545789
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56545829
ē
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
annotationsŠ *
 
+:)@@2conv2d_617/kernel
(
Þ0"
trackable_list_wrapper
(
Þ0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ß	variables
ātrainable_variables
áregularization_losses
ã__call__
+ä&call_and_return_all_conditional_losses
'ä"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_617_layer_call_fn_56545836Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56545843Ē
ē
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
annotationsŠ *
 
2:0@2$batch_instance_normalization_509/rho
4:2@2&batch_instance_normalization_509/gamma
3:1@2%batch_instance_normalization_509/beta
8
å0
æ1
į2"
trackable_list_wrapper
8
å0
æ1
į2"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
č	variables
étrainable_variables
ęregularization_losses
ė__call__
+í&call_and_return_all_conditional_losses
'í"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_509_layer_call_fn_56545854
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56545894
ē
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
annotationsŠ *
 
+:)@2conv2d_618/kernel
(
î0"
trackable_list_wrapper
(
î0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ï	variables
ðtrainable_variables
ņregularization_losses
ó__call__
+ô&call_and_return_all_conditional_losses
'ô"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_618_layer_call_fn_56545901Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56545908Ē
ē
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
annotationsŠ *
 
2:02$batch_instance_normalization_510/rho
4:22&batch_instance_normalization_510/gamma
3:12%batch_instance_normalization_510/beta
8
õ0
ö1
ũ2"
trackable_list_wrapper
8
õ0
ö1
ũ2"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ø	variables
ųtrainable_variables
úregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
č2å
C__inference_batch_instance_normalization_510_layer_call_fn_56545919
ē
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
annotationsŠ *
 
2
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56545959
ē
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
annotationsŠ *
 
+:)2conv2d_619/kernel
(
þ0"
trackable_list_wrapper
(
þ0"
trackable_list_wrapper
 "
trackable_list_wrapper
ļ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ĸ	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Ũ2Ô
-__inference_conv2d_619_layer_call_fn_56545966Ē
ē
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
annotationsŠ *
 
ō2ï
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56545973Ē
ē
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
annotationsŠ *
 
 "
trackable_list_wrapper
þ
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
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÔBŅ
&__inference_signature_wrapper_56545074input_1input_2"
ē
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
annotationsŠ *
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
trackable_dict_wrapper§
#__inference__wrapped_model_56541693ĸP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþlĒi
bĒ_
]Z
+(
input_1ĸĸĸĸĸĸĸĸĸ
+(
input_2ĸĸĸĸĸĸĸĸĸ
Š "=Š:
8
output_1,)
output_1ĸĸĸĸĸĸĸĸĸĖ
^__inference_batch_instance_normalization_500_layer_call_and_return_conditional_losses_56545177jABC3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ@@
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ@@
 Ī
C__inference_batch_instance_normalization_500_layer_call_fn_56545137]ABC3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ@@
Š "!ĸĸĸĸĸĸĸĸĸ@@Ė
^__inference_batch_instance_normalization_501_layer_call_and_return_conditional_losses_56545242jQRS3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ@@
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ@@
 Ī
C__inference_batch_instance_normalization_501_layer_call_fn_56545202]QRS3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ@@
Š "!ĸĸĸĸĸĸĸĸĸ@@Ė
^__inference_batch_instance_normalization_502_layer_call_and_return_conditional_losses_56545329jghi3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ  
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ  
 Ī
C__inference_batch_instance_normalization_502_layer_call_fn_56545289]ghi3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ  
Š "!ĸĸĸĸĸĸĸĸĸ  Ė
^__inference_batch_instance_normalization_503_layer_call_and_return_conditional_losses_56545406jwxy3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ  
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ  
 Ī
C__inference_batch_instance_normalization_503_layer_call_fn_56545366]wxy3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ  
Š "!ĸĸĸĸĸĸĸĸĸ  Ï
^__inference_batch_instance_normalization_504_layer_call_and_return_conditional_losses_56545483m3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ  
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ  
 §
C__inference_batch_instance_normalization_504_layer_call_fn_56545443`3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ  
Š "!ĸĸĸĸĸĸĸĸĸ  Ï
^__inference_batch_instance_normalization_505_layer_call_and_return_conditional_losses_56545560m3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ  
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ  
 §
C__inference_batch_instance_normalization_505_layer_call_fn_56545520`3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ  
Š "!ĸĸĸĸĸĸĸĸĸ  Ï
^__inference_batch_instance_normalization_506_layer_call_and_return_conditional_losses_56545662mŪŊ°3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ@@
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ@@
 §
C__inference_batch_instance_normalization_506_layer_call_fn_56545622`ŪŊ°3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ@@
Š "!ĸĸĸĸĸĸĸĸĸ@@Ï
^__inference_batch_instance_normalization_507_layer_call_and_return_conditional_losses_56545727mūŋĀ3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ@@
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ@@
 §
C__inference_batch_instance_normalization_507_layer_call_fn_56545687`ūŋĀ3Ē0
)Ē&
$!
xĸĸĸĸĸĸĸĸĸ@@
Š "!ĸĸĸĸĸĸĸĸĸ@@Ņ
^__inference_batch_instance_normalization_508_layer_call_and_return_conditional_losses_56545829oÕÖŨ4Ē1
*Ē'
%"
xĸĸĸĸĸĸĸĸĸ@
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ@
 Đ
C__inference_batch_instance_normalization_508_layer_call_fn_56545789bÕÖŨ4Ē1
*Ē'
%"
xĸĸĸĸĸĸĸĸĸ@
Š ""ĸĸĸĸĸĸĸĸĸ@Ņ
^__inference_batch_instance_normalization_509_layer_call_and_return_conditional_losses_56545894oåæį4Ē1
*Ē'
%"
xĸĸĸĸĸĸĸĸĸ@
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ@
 Đ
C__inference_batch_instance_normalization_509_layer_call_fn_56545854båæį4Ē1
*Ē'
%"
xĸĸĸĸĸĸĸĸĸ@
Š ""ĸĸĸĸĸĸĸĸĸ@Ņ
^__inference_batch_instance_normalization_510_layer_call_and_return_conditional_losses_56545959oõöũ4Ē1
*Ē'
%"
xĸĸĸĸĸĸĸĸĸ
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ
 Đ
C__inference_batch_instance_normalization_510_layer_call_fn_56545919bõöũ4Ē1
*Ē'
%"
xĸĸĸĸĸĸĸĸĸ
Š ""ĸĸĸĸĸĸĸĸĸŧ
H__inference_conv2d_606_layer_call_and_return_conditional_losses_56545088o&9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ@
 
-__inference_conv2d_606_layer_call_fn_56545081b&9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ
Š ""ĸĸĸĸĸĸĸĸĸ@ŧ
H__inference_conv2d_607_layer_call_and_return_conditional_losses_56545102o-9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ@
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ@
 
-__inference_conv2d_607_layer_call_fn_56545095b-9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ@
Š ""ĸĸĸĸĸĸĸĸĸ@ļ
H__inference_conv2d_608_layer_call_and_return_conditional_losses_56545126l:7Ē4
-Ē*
(%
inputsĸĸĸĸĸĸĸĸĸ@@@
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ@@
 
-__inference_conv2d_608_layer_call_fn_56545119_:7Ē4
-Ē*
(%
inputsĸĸĸĸĸĸĸĸĸ@@@
Š "!ĸĸĸĸĸĸĸĸĸ@@đ
H__inference_conv2d_609_layer_call_and_return_conditional_losses_56545191mJ8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ@@
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ@@
 
-__inference_conv2d_609_layer_call_fn_56545184`J8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ@@
Š "!ĸĸĸĸĸĸĸĸĸ@@đ
H__inference_conv2d_610_layer_call_and_return_conditional_losses_56545278m`8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ  
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ  
 
-__inference_conv2d_610_layer_call_fn_56545259``8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ  
Š "!ĸĸĸĸĸĸĸĸĸ  đ
H__inference_conv2d_611_layer_call_and_return_conditional_losses_56545355mp8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ  
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ  
 
-__inference_conv2d_611_layer_call_fn_56545336`p8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ  
Š "!ĸĸĸĸĸĸĸĸĸ  š
H__inference_conv2d_612_layer_call_and_return_conditional_losses_56545432n8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ  
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ  
 
-__inference_conv2d_612_layer_call_fn_56545413a8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ  
Š "!ĸĸĸĸĸĸĸĸĸ  š
H__inference_conv2d_613_layer_call_and_return_conditional_losses_56545509n8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ  
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ  
 
-__inference_conv2d_613_layer_call_fn_56545490a8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ  
Š "!ĸĸĸĸĸĸĸĸĸ  š
H__inference_conv2d_614_layer_call_and_return_conditional_losses_56545611n§8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ@@
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ@@
 
-__inference_conv2d_614_layer_call_fn_56545604a§8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ@@
Š "!ĸĸĸĸĸĸĸĸĸ@@š
H__inference_conv2d_615_layer_call_and_return_conditional_losses_56545676n·8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ@@
Š ".Ē+
$!
0ĸĸĸĸĸĸĸĸĸ@@
 
-__inference_conv2d_615_layer_call_fn_56545669a·8Ē5
.Ē+
)&
inputsĸĸĸĸĸĸĸĸĸ@@
Š "!ĸĸĸĸĸĸĸĸĸ@@―
H__inference_conv2d_616_layer_call_and_return_conditional_losses_56545778qÎ:Ē7
0Ē-
+(
inputsĸĸĸĸĸĸĸĸĸ
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ@
 
-__inference_conv2d_616_layer_call_fn_56545771dÎ:Ē7
0Ē-
+(
inputsĸĸĸĸĸĸĸĸĸ
Š ""ĸĸĸĸĸĸĸĸĸ@ž
H__inference_conv2d_617_layer_call_and_return_conditional_losses_56545843pÞ9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ@
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ@
 
-__inference_conv2d_617_layer_call_fn_56545836cÞ9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ@
Š ""ĸĸĸĸĸĸĸĸĸ@ž
H__inference_conv2d_618_layer_call_and_return_conditional_losses_56545908pî9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ@
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ
 
-__inference_conv2d_618_layer_call_fn_56545901cî9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ@
Š ""ĸĸĸĸĸĸĸĸĸž
H__inference_conv2d_619_layer_call_and_return_conditional_losses_56545973pþ9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ
 
-__inference_conv2d_619_layer_call_fn_56545966cþ9Ē6
/Ē,
*'
inputsĸĸĸĸĸĸĸĸĸ
Š ""ĸĸĸĸĸĸĸĸĸé
R__inference_conv2d_transpose_100_layer_call_and_return_conditional_losses_56545597 JĒG
@Ē=
;8
inputs,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Š "@Ē=
63
0,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 Á
7__inference_conv2d_transpose_100_layer_call_fn_56545567 JĒG
@Ē=
;8
inputs,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Š "30,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸč
R__inference_conv2d_transpose_101_layer_call_and_return_conditional_losses_56545764ĮJĒG
@Ē=
;8
inputs,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Š "?Ē<
52
0+ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@
 Ā
7__inference_conv2d_transpose_101_layer_call_fn_56545734ĮJĒG
@Ē=
;8
inputs,ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Š "2/+ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ@Á
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543521õP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþpĒm
fĒc
]Z
+(
input_1ĸĸĸĸĸĸĸĸĸ
+(
input_2ĸĸĸĸĸĸĸĸĸ
p 
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ
 Á
G__inference_face_g_18_layer_call_and_return_conditional_losses_56543672õP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþpĒm
fĒc
]Z
+(
input_1ĸĸĸĸĸĸĸĸĸ
+(
input_2ĸĸĸĸĸĸĸĸĸ
p
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ
 Ã
G__inference_face_g_18_layer_call_and_return_conditional_losses_56544424ũP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþrĒo
hĒe
_\
,)
inputs/0ĸĸĸĸĸĸĸĸĸ
,)
inputs/1ĸĸĸĸĸĸĸĸĸ
p 
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ
 Ã
G__inference_face_g_18_layer_call_and_return_conditional_losses_56544968ũP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþrĒo
hĒe
_\
,)
inputs/0ĸĸĸĸĸĸĸĸĸ
,)
inputs/1ĸĸĸĸĸĸĸĸĸ
p
Š "/Ē,
%"
0ĸĸĸĸĸĸĸĸĸ
 
,__inference_face_g_18_layer_call_fn_56542663čP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþpĒm
fĒc
]Z
+(
input_1ĸĸĸĸĸĸĸĸĸ
+(
input_2ĸĸĸĸĸĸĸĸĸ
p 
Š ""ĸĸĸĸĸĸĸĸĸ
,__inference_face_g_18_layer_call_fn_56543370čP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþpĒm
fĒc
]Z
+(
input_1ĸĸĸĸĸĸĸĸĸ
+(
input_2ĸĸĸĸĸĸĸĸĸ
p
Š ""ĸĸĸĸĸĸĸĸĸ
,__inference_face_g_18_layer_call_fn_56543776ęP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþrĒo
hĒe
_\
,)
inputs/0ĸĸĸĸĸĸĸĸĸ
,)
inputs/1ĸĸĸĸĸĸĸĸĸ
p 
Š ""ĸĸĸĸĸĸĸĸĸ
,__inference_face_g_18_layer_call_fn_56543880ęP&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþrĒo
hĒe
_\
,)
inputs/0ĸĸĸĸĸĸĸĸĸ
,)
inputs/1ĸĸĸĸĸĸĸĸĸ
p
Š ""ĸĸĸĸĸĸĸĸĸō
O__inference_max_pooling2d_100_layer_call_and_return_conditional_losses_56545112RĒO
HĒE
C@
inputs4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Š "HĒE
>;
04ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 Ę
4__inference_max_pooling2d_100_layer_call_fn_56545107RĒO
HĒE
C@
inputs4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Š ";84ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸō
O__inference_max_pooling2d_101_layer_call_and_return_conditional_losses_56545252RĒO
HĒE
C@
inputs4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Š "HĒE
>;
04ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
 Ę
4__inference_max_pooling2d_101_layer_call_fn_56545247RĒO
HĒE
C@
inputs4ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸ
Š ";84ĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸĸŧ
&__inference_signature_wrapper_56545074P&-:ABCJQRS`ghipwxy §ŪŊ°·ūŋĀĮÎÕÖŨÞåæįîõöũþ}Ēz
Ē 
sŠp
6
input_1+(
input_1ĸĸĸĸĸĸĸĸĸ
6
input_2+(
input_2ĸĸĸĸĸĸĸĸĸ"=Š:
8
output_1,)
output_1ĸĸĸĸĸĸĸĸĸ