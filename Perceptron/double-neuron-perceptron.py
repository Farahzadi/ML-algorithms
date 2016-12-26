#!
import numpy as np
from colorama import Fore, init
from matplotlib import pyplot as plt


init( autoreset=True )
plt.style.use( 'ggplot' )
plt.rcParams[ 'figure.figsize' ] = [ 9, 9 ]


def random_t_set_generator( ):
	a_set, b_set, c_set, d_set = [ ], [ ], [ ], [ ]
	for i in range( 20 ):
		a_set.append( (np.random.uniform( 0, .5 ), np.random.uniform( 0, .4 ), 0) )  # 00
		b_set.append( (np.random.uniform( 0, .5 ), np.random.uniform( .5, 1 ), 1) )  # 01
		c_set.append( (np.random.uniform( .6, 1 ), np.random.uniform( 0, .4 ), 2) )  # 10
		d_set.append( (np.random.uniform( .6, 1 ), np.random.uniform( .5, 1 ), 3) )  # 11
	training_set = np.vstack( (np.array( a_set ), np.array( b_set ), np.array( c_set ), np.array( d_set )) )
	np.random.shuffle( training_set )
	return a_set, b_set, c_set, d_set, training_set


def draw_plane( weight, bias, a_set, b_set, c_set, d_set, pattern, output, target, iteration ):
	def f( x ):
		return (-bias - weight[ 0 ] * x) / weight[ 1 ]

	plt.xlim( (-.5, 1.5) )
	plt.ylim( (-.5, 1.5) )
	print( )
	plt.annotate( s='weight: {}\n'
						 'bias: {}\n'
						 'pattern: {}\n'
						 'output: {}\n'
						 'target: {}\n'
						 'iteration: {}'.format( weight, bias, pattern, output, target, iteration ), xy=(-.5, 1.5) )
	plt.scatter( [ p[ 0 ] for p in a_set ], [ p[ 1 ] for p in a_set ], marker='o', c='#f44336' )
	plt.scatter( [ p[ 0 ] for p in b_set ], [ p[ 1 ] for p in b_set ], marker='x', c='#673AB7' )
	plt.plot( [ 0, weight[ 0 ] ], [ 0, weight[ 1 ] ], c="red", label='weight vector' )
	plt.plot( [ -1, 2 ], [ f( -1 ), f( 2 ) ], c='black', label='decision boundary' )
	plt.legend( )
	plt.show( )


def hardlim( x ):
	return 1 if x >= 0 else 0


def train( training_set, a_set, b_set, c_set, d_set ):
	iteration = 0
	answer_iteration = 0
	c_count = 0
	t_count = training_set.shape[ 0 ]
	weight = np.random.uniform( .1, 1, (2,) )
	bias = np.random.rand( )
	print( Fore.CYAN + 'Initial weight: {}\n'
							 'Initial bias: {}\n'.format( weight, bias ) )
	print( '-' * 120 )
	while True:
		for item in training_set:
			iteration += 1
			p = np.array( [ item[ 0 ], item[ 1 ] ] )
			t = item[ 2 ]
			a = hardlim( weight.dot( p ) + bias )
			error = t - a
			draw_plane( weight, bias, a_set, b_set, p, a, t, iteration )
			if error:
				answer_iteration = 0
				c_count = 0
				weight += error * p
				bias += error
			else:
				if not c_count:
					answer_iteration = iteration
				c_count += 1
			if iteration > t_count and c_count >= 10:
				print( Fore.GREEN + 'iteration: {}\n'
										  'weight: {}\n'
										  'bias: {}\n'
										  'pattern: {}\n'
										  'output: {}\n'
										  'target: {}\n'.format( answer_iteration, weight, bias, p, a, t ) )
				print( Fore.GREEN + 'training was successful!' )
				return
		if iteration > 1000:
			print( Fore.RED + 'training failed!' )
			return


def main( ):
	plt.xlim( (-.5, 1.5) )
	plt.ylim( (-.5, 1.5) )
	a_set, b_set, c_set, d_set, training_set = random_t_set_generator( )
	plt.scatter( [ p[ 0 ] for p in a_set ], [ p[ 1 ] for p in a_set ], marker='o', c='#f44336' )
	plt.scatter( [ p[ 0 ] for p in b_set ], [ p[ 1 ] for p in b_set ], marker='v', c='#EC407A' )
	plt.scatter( [ p[ 0 ] for p in c_set ], [ p[ 1 ] for p in c_set ], marker='^', c='#009688' )
	plt.scatter( [ p[ 0 ] for p in d_set ], [ p[ 1 ] for p in d_set ], marker='s', c='#0D47A1' )
	plt.show( )


# input( )
# train( training_set, a_set, b_set )


if __name__ == '__main__':
	main( )
