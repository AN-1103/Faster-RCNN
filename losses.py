import tensorflow as tf
from keras import backend as K
from keras.objectives import categorical_crossentropy

# if K.image_dim_ordering() == 'tf':
if K.image_data_format() == "channels_first":
	import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

#RPN回归损失 用了smooth l1损失函数
#y_true, 实际上是之前在calc_rpn中计算出来的原图GTbox和anchor的平移缩放因子t，而y_pred是rpn网络计算出来的这个t_pred。
def rpn_loss_regr(num_anchors):
	def rpn_loss_regr_fixed_num(y_true, y_pred):
		# 维度顺序不一样
		# if K.image_dim_ordering() == 'th':  旧keras版本使用此行代码
		if K.image_data_format() == "channels_first":
			x = y_true[:, 4 * num_anchors:, :, :] - y_pred
			x_abs = K.abs(x)
			x_bool = K.less_equal(x_abs, 1.0)
			return lambda_rpn_regr * K.sum(
				y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
		else:
			# y_true 维度是(None, 高,宽，72) 其中72里面前36位是9个锚框对应的回归梯度的有效位，其实就是是不是物体，
			# 是物体取1，不是取0，后36位才是回归梯度
			# 所以求和就是把所是物体的回归梯度值加起来，然后除以是物体的个数求平均
			# 真实和预测之间的回归梯度值的差
			x = y_true[:, :, :, 4 * num_anchors:] - y_pred
			# 取绝对值
			x_abs = K.abs(x)
			# 判断每个元素是否小于1
			x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
			# 乘以这个y_true[:, :, :, :4 * num_anchors]是标志位，表示这个值是不是物体的，值是0或者1，表示只计算有物体是1，没有物体的其实是0
			#因为使用了Smooth L1误差函数,所以才有绝对值判断，小于1判断，这样做使得损失对于误差小的时候变换不敏感
			return lambda_rpn_regr * K.sum(
				y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

	return rpn_loss_regr_fixed_num


# RPN 分类损失，二分类是否有物体
def rpn_loss_cls(num_anchors):
	def rpn_loss_cls_fixed_num(y_true, y_pred):
		if K.image_dim_ordering() == 'tf':
			# 分类的时候只需要考虑有效的标签，也就是前面9位，后面9位是具体分类值，是不是物体，用于交叉熵计算
			# 这个就是二分类交叉熵损失，这里的y_true前9位也是有效位，其实就是只算正负样本，不算中立的样本，后就9位才是真的分类，所以可以求和来求个数。
			return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
		else:
			return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])

	return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
	def class_loss_regr_fixed_num(y_true, y_pred):
		x = y_true[:, :, 4*num_classes:] - y_pred
		x_abs = K.abs(x)
		x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
		return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
	return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
	return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
