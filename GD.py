import glob
from PIL import Image
import math
import numpy as np
from numpy import linalg as la
from numpy.core.getlimits import inf

class GD:
	# Perform gradient descent.
	# The objective_function must takes in a matrix and compute the value and
	# gradient with respect to the input. w is a matrix, with initial guesses.
	# epsilon is the gradient that would be considered to be close enough to a minima.
	# turning_angle is the max angle in radian that consecutive descending steps
	# are allowed to have, 0 up to pi.
	# past_gradient_ratio is to be used when performing stochastic gradient
	# descent which requires objective_function to contain such implementation.
	# It is between 0 and 1 where 0 represent to only use new gradient. It is
	# to be used to balance out the randomness when using stochastic gradient
	# descent.
	# It will perform at most max_iteration cycles of gradient descent.
	# Returns a tuple of resulted w and a boolean. True if it is able find 
	# a point that is close enough to a minima, false otherwise.
	def gradient_descent(objective_function, w, turning_angle=math.pi, past_gradient_ratio=0, epsilon=1e-3, max_iteration=50):
		w_prev = w
		f_prev, g_prev = objective_function(w)
		alpha = min(1.0, 1.0/np.sum(np.abs(g_prev)))
		angle = math.cos(turning_angle)
		if past_gradient_ratio != 0:
			past_gradient = g_prev
		g_bearing = 2

		for i in range(max_iteration):
			w_next = w_prev - alpha*g_prev
			f_next, g_next = objective_function(w_next)
			if past_gradient_ratio != 0:
				g_next = (1-past_gradient_ratio)*g_next + past_gradient_ratio*past_gradient
			if turning_angle != math.pi:
				g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))				

			while f_next > f_prev or g_bearing < angle:
				alpha_new = alpha / 2
				if alpha_new == 0:
					return (w_prev, True)
				alpha = alpha_new
				w_next = w_prev - alpha*g_prev
				f_next, g_next = objective_function(w_next)
				if past_gradient_ratio != 0:
					g_next = (1-past_gradient_ratio)*g_next + past_gradient_ratio*past_gradient
				if turning_angle != math.pi:
					g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			w_prev = w_next
			f_prev = f_next
			g_prev = g_next

			if la.norm(g_prev) <= epsilon:
				return (w_prev, True)
		
		return (w_prev, False)
	
	# Perform gradient descent with the constraint that the result is non-negative.
	# The objective_function must takes in a matrix and compute the value and
	# gradient with respect to the input. w is a matrix, with initial guesses.
	# epsilon is the gradient that would be considered to be close enough to a minima.
	# turning_angle is the max angle in radian that consecutive descending steps
	# are allowed to have, 0 up to pi.
	# It will perform at most max_iteration cycles of gradient descent.
	# Returns a tuple of resulted w and a boolean. True if it is able find 
	# a point that is close enough to a minima, false otherwise.
	def gradient_descent_non_negative(objective_function, w, turning_angle=math.pi, epsilon=1e-2, max_iteration=30):
		w_prev = np.maximum(w, 0)
		f_prev, g_prev = objective_function(w)
		alpha = min(1.0, 1.0/np.sum(np.abs(g_prev)))
		angle = math.cos(turning_angle)

		for i in range(max_iteration):
			w_next = w_prev - alpha*g_prev
			w_next = np.maximum(w_next, 0)
			f_next, g_next = objective_function(w_next)
			g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			while f_next > f_prev or g_bearing < angle:
				alpha_new = alpha / 2
				if alpha_new == 0:
					return (w_prev, True)
				alpha = alpha_new
				w_next = w_prev - alpha*g_prev
				w_next = np.maximum(w_next, 0)
				f_next, g_next = objective_function(w_next)
				g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			w_prev = w_next
			f_prev = f_next
			g_prev = g_next

			if la.norm(g_prev) <= epsilon:
				return (w_prev, True)
		
		return (w_prev, False)

	# Perform gradient descent with l1 regularization.
	# The objective_function must takes in a matrix and compute the value and
	# gradient with respect to the input. w is a matrix, with initial guesses.
	# epsilon is the gradient that would be considered to be close enough to a minima.
	# turning_angle is the max angle in radian that consecutive descending steps
	# are allowed to have, 0 up to pi.
	# It will perform at most max_iteration cycles of gradient descent.
	# Returns a tuple of resulted w and a boolean. True if it is able find 
	# a point that is close enough to a minima, false otherwise.
	def gradient_descent_l1(objective_function, w, l=1, turning_angle=math.pi, epsilon=1e-2, max_iteration=30):
		w_prev = w
		f_prev, g_prev = objective_function(w)
		alpha = min(1.0, 1.0/np.sum(np.abs(g_prev)))
		angle = math.cos(turning_angle)

		for i in range(max_iteration):
			w_next = w_prev - alpha*g_prev
			w_next = np.multiply(np.sign(w_next), np.maximum(np.abs(w_next) - alpha*l, 0))
			f_next, g_next = objective_function(w_next)
			g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			while f_next > f_prev or g_bearing < angle:
				alpha_new = alpha / 2
				if alpha_new == 0:
					return (w_prev, True)
				alpha = alpha_new
				w_next = w_prev - alpha*g_prev
				w_next = np.multiply(np.sign(w_next), np.maximum(np.abs(w_next) - alpha*l, 0))
				f_next, g_next = objective_function(w_next)
				g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			w_prev = w_next
			f_prev = f_next
			g_prev = g_next

			if la.norm(g_prev) <= epsilon:
				return (w_prev, True)
		
		return (w_prev, False)
	
	# Perform gradient descent with the constraint that the result is non-negative
	# and with l1 regularization.
	# The objective_function must takes in a matrix and compute the value and
	# gradient with respect to the input. w is a matrix, with initial guesses.
	# epsilon is the gradient that would be considered to be close enough to a minima.
	# turning_angle is the max angle in radian that consecutive descending steps
	# are allowed to have, 0 up to pi.
	# It will perform at most max_iteration cycles of gradient descent.
	# Returns a tuple of resulted w and a boolean. True if it is able find 
	# a point that is close enough to a minima, false otherwise.
	def gradient_descent_non_negative_l1(objective_function, w, l=1, turning_angle=math.pi, epsilon=1e-2, max_iteration=30):
		w_prev = np.maximum(w, 0)
		f_prev, g_prev = objective_function(w)
		alpha = min(1.0, 1.0/np.sum(np.abs(g_prev)))
		angle = math.cos(turning_angle)

		for i in range(max_iteration):
			w_next = w_prev - alpha*g_prev
			w_next = np.multiply(np.sign(w_next), np.maximum(np.abs(w_next) - alpha*l, 0))
			w_next = np.maximum(w_next, 0)
			f_next, g_next = objective_function(w_next)
			g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			while f_next > f_prev or g_bearing < angle:
				alpha_new = alpha / 2
				if alpha_new == 0:
					return (w_prev, True)
				alpha = alpha_new
				w_next = w_prev - alpha*g_prev
				w_next = np.multiply(np.sign(w_next), np.maximum(np.abs(w_next) - alpha*l, 0))
				w_next = np.maximum(w_next, 0)
				f_next, g_next = objective_function(w_next)
				g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			w_prev = w_next
			f_prev = f_next
			g_prev = g_next

			if la.norm(g_prev) <= epsilon:
				return (w_prev, True)
		
		return (w_prev, False)
	
	# Perform gradient descent with l2 regularization.
	# The objective_function must takes in a matrix and compute the value and
	# gradient with respect to the input. w is a matrix, with initial guesses.
	# epsilon is the gradient that would be considered to be close enough to a minima.
	# turning_angle is the max angle in radian that consecutive descending steps
	# are allowed to have, 0 up to pi.
	# It will perform at most max_iteration cycles of gradient descent.
	# Returns a tuple of resulted w and a boolean. True if it is able find 
	# a point that is close enough to a minima, false otherwise.
	def gradient_descent_l2(objective_function, w, l=1, turning_angle=math.pi, epsilon=1e-2, max_iteration=30):
		w_prev = w
		f_prev, g_prev = objective_function(w)
		alpha = min(1.0, 1.0/np.sum(np.abs(g_prev)))
		angle = math.cos(turning_angle)

		for i in range(max_iteration):
			w_next = w_prev - alpha*(g_prev + l*w_prev)
			f_next, g_next = objective_function(w_next)
			g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			while f_next > f_prev or g_bearing < angle:
				alpha_new = alpha / 2
				if alpha_new == 0:
					return (w_prev, True)
				alpha = alpha_new
				w_next = w_prev - alpha*(g_prev + l*w_prev)
				f_next, g_next = objective_function(w_next)
				g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			w_prev = w_next
			f_prev = f_next
			g_prev = g_next

			if la.norm(g_prev) <= epsilon:
				return (w_prev, True)
		
		return (w_prev, False)
	
	# Perform gradient descent with the constraint that the result is non-negative
	# and with l2 regularization.
	# The objective_function must takes in a matrix and compute the value and
	# gradient with respect to the input. w is a matrix, with initial guesses.
	# epsilon is the gradient that would be considered to be close enough to a minima.
	# turning_angle is the max angle in radian that consecutive descending steps
	# are allowed to have, 0 up to pi.
	# It will perform at most max_iteration cycles of gradient descent.
	# Returns a tuple of resulted w and a boolean. True if it is able find 
	# a point that is close enough to a minima, false otherwise.
	def gradient_descent_non_negative_l2(objective_function, w, l=1, turning_angle=math.pi, epsilon=1e-2, max_iteration=30):
		w_prev = np.maximum(w, 0)
		f_prev, g_prev = objective_function(w)
		alpha = min(1.0, 1.0/np.sum(np.abs(g_prev)))
		angle = math.cos(turning_angle)

		for i in range(max_iteration):
			w_next = w_prev - alpha*(g_prev + l*w_prev)
			w_next = np.maximum(w_next, 0)
			f_next, g_next = objective_function(w_next)
			g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			while f_next > f_prev or g_bearing < angle:
				alpha_new = alpha / 2
				if alpha_new == 0:
					return (w_prev, True)
				alpha = alpha_new
				w_next = w_prev - alpha*(g_prev + l*w_prev)
				w_next = np.maximum(w_next, 0)
				f_next, g_next = objective_function(w_next)
				g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			w_prev = w_next
			f_prev = f_next
			g_prev = g_next

			if la.norm(g_prev) <= epsilon:
				return (w_prev, True)
		
		return (w_prev, False)
	
	# Perform gradient descent with l-infinity regularization.
	# The objective_function must takes in a matrix and compute the value and
	# gradient with respect to the input. w is a matrix, with initial guesses.
	# epsilon is the gradient that would be considered to be close enough to a minima.
	# turning_angle is the max angle in radian that consecutive descending steps
	# are allowed to have, 0 up to pi.
	# It will perform at most max_iteration cycles of gradient descent.
	# Returns a tuple of resulted w and a boolean. True if it is able find 
	# a point that is close enough to a minima, false otherwise.
	def gradient_descent_linfinity(objective_function, w, l=1, turning_angle=math.pi, epsilon=1e-2, max_iteration=30):
		w_prev = w
		f_prev, g_prev = objective_function(w)
		alpha = min(1.0, 1.0/np.sum(np.abs(g_prev)))
		angle = math.cos(turning_angle)

		for i in range(max_iteration):
			w_next = w_prev - alpha*(g_prev + l*max(w_prev.max(), w_prev.min(), key=abs))
			f_next, g_next = objective_function(w_next)
			g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			while f_next > f_prev or g_bearing < angle:
				alpha_new = alpha / 2
				if alpha_new == 0:
					return (w_prev, True)
				alpha = alpha_new
				w_next = w_prev - alpha*(g_prev + l*max(w_prev.max(), w_prev.min(), key=abs))
				f_next, g_next = objective_function(w_next)
				g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			w_prev = w_next
			f_prev = f_next
			g_prev = g_next

			if la.norm(g_prev) <= epsilon:
				return (w_prev, True)
		
		return (w_prev, False)
	
	# Perform gradient descent with the constraint that the result is non-negative
	# and with l-infinity regularization.
	# The objective_function must takes in a matrix and compute the value and
	# gradient with respect to the input. w is a matrix, with initial guesses.
	# epsilon is the gradient that would be considered to be close enough to a minima.
	# turning_angle is the max angle in radian that consecutive descending steps
	# are allowed to have, 0 up to pi.
	# It will perform at most max_iteration cycles of gradient descent.
	# Returns a tuple of resulted w and a boolean. True if it is able find 
	# a point that is close enough to a minima, false otherwise.
	def gradient_descent_non_negative_linfinity(objective_function, w, l=1, turning_angle=math.pi, epsilon=1e-3, max_iteration=30):
		w_prev = np.maximum(w, 0)
		f_prev, g_prev = objective_function(w)
		alpha = min(1.0, 1.0/np.sum(np.abs(g_prev)))
		angle = math.cos(turning_angle)

		for i in range(max_iteration):
			w_next = w_prev - alpha*(g_prev + l*max(w_prev.max(), w_prev.min(), key=abs))
			w_next = np.maximum(w_next, 0)
			f_next, g_next = objective_function(w_next)
			g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			while f_next > f_prev or g_bearing < angle:
				alpha_new = alpha / 2
				if alpha_new == 0:
					return (w_prev, True)
				alpha = alpha_new
				w_next = w_prev - alpha*(g_prev + l*max(w_prev.max(), w_prev.min(), key=abs))
				w_next = np.maximum(w_next, 0)
				f_next, g_next = objective_function(w_next)
				g_bearing = np.sum(np.multiply(g_prev, g_next)) / (la.norm(g_prev)*la.norm(g_next))

			w_prev = w_next
			f_prev = f_next
			g_prev = g_next

			if la.norm(g_prev) <= epsilon:
				return (w_prev, True)
		
		return (w_prev, False)

# Principal Component Analysis
# factorize matrix X (nxd) into Z (nxk, features) and
# W (kxd, principal component).
# Note: principal components have the same number of features as X.
# The objective is to optimize the product of Z and W to be as close to X as
# possible. By definition, it is using the l2-loss, the squared differences,
# while other losses are possible.
# Namely, f = 1/2||ZW-X||^2
# L1-regularization is used to produce more sparse result that are easier to
# interpret. It also produces better result due to less over-fitting.
# Alternating optimization of Z and W would converge to a global minima given
# random initialization.
class PCA:
	epsilon = 1e-2
	max_iteration = 100

	def __init__(self, X: np.matrix):
		self.X = X
		self.n, self.d = X.shape

	# In this case of l2-loss, the derivative of the function happens to be R,
	# the residual, and the dR used during computing G is the derivative of f with
	# respect to R.
	def get_value_and_gradient_wrt_z(self, Z):
		dR = R = Z*self.W - self.X
		f = (1/2)*np.sum(np.power(R, 2))
		G = dR*np.transpose(self.W)
		return f, G

	# In this case of l2-loss, the derivative of the function happens to be R,
	# the residual, and the dR used during computing G is the derivative of f with
	# respect to R.
	def get_stochastic_value_and_gradient_wrt_z(self, Z):
		index = np.random.randint(self.k)
		W = self.W[:,index]
		X = self.X[:,index]
		dR = R = Z*W - X
		f = (1/2)*np.sum(np.power(R, 2))
		G = dR*np.transpose(W)
		return f, G

	# In this case of l2-loss, the derivative of the function happens to be R,
	# the residual, and the dR used during computing G is the derivative of f with
	# respect to R.
	def get_value_and_gradient_wrt_w(self, W):
		dR = R = self.Z*W - self.X
		f = (1/2)*np.sum(np.power(R, 2))
		G = np.transpose(self.Z)*dR
		return f, G

	# In this case of l2-loss, the derivative of the function happens to be R,
	# the residual, and the dR used during computing G is the derivative of f with
	# respect to R.
	def get_stochastic_value_and_gradient_wrt_w(self, W):
		index = np.random.randint(self.n)
		Z = self.Z[index,:]
		X = self.X[index,:]
		dR = R = Z*W - X
		f = (1/2)*np.sum(np.power(R, 2))
		G = np.transpose(Z)*dR
		return f, G

	def initialize_z_and_w(self, k: int):
		self.Z = np.matrix(np.matrix(np.random.rand(self.n, k)))
		self.W = np.matrix(np.matrix(np.random.rand(k, self.d)))

	# It is important to center the data, otherwise it is lossing accuracy
	# due to assuming that the origin is part of the result.
	def regularized_PCA(self, k: int, l=0):
		self.k = k
		self.mean = self.X.mean()
		self.X -= self.mean
		self.initialize_z_and_w(k)
		f_previous = np.inf
		for i in range(self.max_iteration):
			self.Z, signal = GD.gradient_descent(self.get_value_and_gradient_wrt_z,
																					
																self.Z,
																turning_angle=math.pi/2)

			self.W, signal = GD.gradient_descent(self.get_value_and_gradient_wrt_w,
																self.W,
																turning_angle=math.pi/2)

			R = self.Z*self.W - self.X
			f = (1/2)*np.sum(np.power(R, 2))
			
			if (f_previous - f)/(self.n*self.d) < self.epsilon:
				break
			
			f_previous = f
		self.X += self.mean

			# It is important to center the data, otherwise it is lossing accuracy
	# due to assuming that the origin is part of the result.
	def stochastic_regularized_PCA(self, k: int, l=0):
		self.k = k
		self.mean = self.X.mean()
		self.X -= self.mean
		self.initialize_z_and_w(k)
		f_previous = np.inf
		for i in range(self.max_iteration):
			self.Z, signal = GD.gradient_descent(self.get_stochastic_value_and_gradient_wrt_z,	
																self.Z,
																past_gradient_ratio=0.95)

			self.W, signal = GD.gradient_descent(self.get_stochastic_value_and_gradient_wrt_w,
																self.W,
																past_gradient_ratio=0.95)

			R = self.Z*self.W - self.X
			f = (1/2)*np.sum(np.power(R, 2))
			
			if (f_previous - f)/(self.n*self.d) < self.epsilon:
				break
			
			f_previous = f
		self.X += self.mean
			
	# The data cannot be centered in this case as all negative values are
	# eliminated. It is trading accuracy with the ability to interpret.
	def regularized_PCA_non_negative_matrix_factorization(self, k: int):
		self.initialize_z_and_w(k)
		f_previous = np.inf
		for i in range(self.max_iteration):
			self.Z, signal = GD.gradient_descent_non_negative(self.get_value_and_gradient_wrt_z,
																self.Z)

			self.W, signal = GD.gradient_descent_non_negative(self.get_value_and_gradient_wrt_w,
																self.W)

			R = self.Z*self.W - self.X
			f = (1/2)*np.sum(np.power(R, 2))
			
			if (f_previous - f)/(self.n*self.d) < self.epsilon:
				break
			
			f_previous = f

images = []
for f in glob.glob("*.jpg"):
    images.append(np.asarray(np.reshape(Image.open(f).convert('L'), -1), np.float64))

images = np.matrix(images)
a=PCA(images)
a.stochastic_regularized_PCA(4)

#a=PCA(np.matrix([[1,2,3,11,12,13],[4,5,6,14,15,16],[7,8,9,17,18,19],[0,0,0,0,0,0],[1,1,1,1,1,1],[2,2,2,2,2,2]]).astype(np.float64))
#a=PCA(np.matrix(np.random.rand(10000,10000)))
#a.stochastic_regularized_PCA(4)
#print((a.Z*a.W+a.mean).round().astype(int))
