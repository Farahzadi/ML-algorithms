#!
import numpy as np
from matplotlib import pyplot as plt
from colorama import init, Fore


init( autoreset=True )


def random_t_set_generator( ):
	a_set, b_set = [ ], [ ]
	for i in range( 20 ):
		a_set.append( (np.random.uniform( 0, .5 ), np.random.uniform( 0, .5 ), 0) )
		b_set.append( (np.random.uniform( .5, 1 ), np.random.uniform( .5, 1 ), 1) )
	training_set = np.vstack( (np.array( a_set ), np.array( b_set )) )
	np.random.shuffle( training_set )
	return a_set, b_set, training_set


def draw_decision_boundary( weight, bias, a_set, b_set ):
	def f( x ):
		return (-bias - weight[ 0 ] * x) / -weight[ 1 ]

	plt.xlim((0, 1))
	plt.ylim( (0, 1) )
	plt.scatter( [ p[ 0 ] for p in a_set ], [ p[ 1 ] for p in a_set ], marker='o', c='blue' )
	plt.scatter( [ p[ 0 ] for p in b_set ], [ p[ 1 ] for p in b_set ], marker='x', c='green' )
	plt.show( )
	plt.plot( [ 0, 1 ], [ f( 0 ), f( 1 ) ] )


def hardlim( x ):
	return 1 if x >= 0 else 0


def train( training_set, a_set, b_set ):
	iteration = 0
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
			draw_decision_boundary( weight, bias, a_set, b_set )
			print( Fore.RED + str( error ) )
			if error:
				c_count = 0
				weight += error * p
			else:
				c_count += 1
			print( Fore.YELLOW + str( c_count ) )
			print( 'iteration: {}\n'
					 'weight: {}\n'
					 'bias: {}\n'
					 'pattern: {}\n'
					 'output: {}\n'
					 'target: {}\n'.format( iteration, weight, bias, p, a, t ) )
			print( '-' * 120 )
			draw_decision_boundary( weight, bias, a_set, b_set )
			if iteration > t_count and c_count >= 5:
				print( Fore.GREEN + 'iteration: {}\n'
						 'weight: {}\n'
						 'bias: {}\n'
						 'pattern: {}\n'
						 'output: {}\n'
						 'target: {}\n'.format( iteration, weight, bias, p, a, t ) )
				print( Fore.GREEN + 'training was successful!' )
				return
		if iteration > 1000:
			print( Fore.RED + 'training failed!' )


def main( ):
	plt.style.use( 'ggplot' )
	plt.xlim( (0, 1) )
	plt.ylim( (0, 1) )
	a_set, b_set, training_set = random_t_set_generator( )
	plt.scatter( [ p[ 0 ] for p in a_set ], [ p[ 1 ] for p in a_set ], marker='o', c='blue' )
	plt.scatter( [ p[ 0 ] for p in b_set ], [ p[ 1 ] for p in b_set ], marker='x', c='green' )
	plt.show( )
	train( training_set, a_set, b_set )


if __name__ == '__main__':
	main( )
