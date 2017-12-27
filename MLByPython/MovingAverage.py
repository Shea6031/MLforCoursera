import itertools
from itertools import islice
from collections import deque


#########################################################
def movingaverage(data, subset_size, data_is_list = None,
		avoid_fp_drift = True):
	
	if subset_size < 1:
		raise ValueError('subset_size must be 1 or larger')

	if data_is_list is None:
		data_is_list = hasattr(data, '__getslice__')

	divisor = float(subset_size)
	if data_is_list:
		#  This only works if we can re-access old elements, but is much faster.
		#  In other words, it can't be just an iterable, it needs to be a list.

		if subset_size > len(data):
			raise ValueError('subset_size must be smaller than data set size')

		if avoid_fp_drift:
			for x in range(subset_size, len(data) + 1):
				yield sum(data[x - subset_size:x]) / divisor
		else:
			cur = sum(data[0:subset_size])
			yield cur / divisor
			for x in range(subset_size, len(data)):
				cur += data[x] - data[x - subset_size]
				yield cur / divisor
	else:
		#  Based on the recipe at:
		#     http://docs.python.org/library/collections.html#deque-recipes
		it = iter(data)
		d = deque(islice(it, subset_size))

		if subset_size > len(d):
			raise ValueError('subset_size must be smaller than data set size')

		if avoid_fp_drift:
			yield sum(d) / divisor
			for elem in it:
				d.popleft()
				d.append(elem)
				yield sum(d) / divisor
		else:
			s = sum(d)
			yield s / divisor
			for elem in it:
				s += elem - d.popleft()
				d.append(elem)
				yield s / divisor


##########################
if __name__ == '__main__':
	import unittest

	class TestMovingAverage(unittest.TestCase):
		####################
		def test_List(self):
			try:
				list(movingaverage([1,2,3], 0))
				self.fail('Did not raise ValueError on subset_size=0')
			except ValueError:
				pass

			try:
				list(movingaverage([1,2,3,4,5,6], 7))
				self.fail('Did not raise ValueError on subset_size > len(data)')
			except ValueError:
				pass

			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 1)), [1,2,3,4,5,6])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 2)),
					[1.5,2.5,3.5,4.5,5.5])
			self.assertEqual(list(movingaverage(map(float, [1,2,3,4,5,6]), 2)),
					[1.5,2.5,3.5,4.5,5.5])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 3)), [2,3,4,5])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 4)), [2.5,3.5,4.5])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 5)), [3,4])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 6)), [3.5])

			self.assertEqual(list(movingaverage([40, 30, 50, 46, 39, 44],
					3, False)), [40.0,42.0,45.0,43.0])
			self.assertEqual(list(movingaverage([40, 30, 50, 46, 39, 44],
					3, True)), [40.0,42.0,45.0,43.0])


		######################
		def test_XRange(self):
			try:
				list(movingaverage(range(1, 4), 0))
				self.fail('Did not raise ValueError on subset_size=0')
			except ValueError:
				pass

			try:
				list(movingaverage(range(1, 7), 7))
				self.fail('Did not raise ValueError on subset_size > len(data)')
			except ValueError:
				pass

			self.assertEqual(list(movingaverage(range(1, 7), 1)), [1,2,3,4,5,6])
			self.assertEqual(list(movingaverage(range(1, 7), 2)),
					[1.5,2.5,3.5,4.5,5.5])
			self.assertEqual(list(movingaverage(iter(map(float, range(1, 7))),
					2)), [1.5,2.5,3.5,4.5,5.5])
			self.assertEqual(list(movingaverage(range(1, 7), 3)), [2,3,4,5])
			self.assertEqual(list(movingaverage(range(1, 7), 4)), [2.5,3.5,4.5])
			self.assertEqual(list(movingaverage(range(1, 7), 5)), [3,4])
			self.assertEqual(list(movingaverage(range(1, 7), 6)), [3.5])


		###########################
		def test_ListRolling(self):
			try:
				list(movingaverage([1,2,3], 0, avoid_fp_drift = False))
				self.fail('Did not raise ValueError on subset_size=0')
			except ValueError:
				pass

			try:
				list(movingaverage([1,2,3,4,5,6], 7, avoid_fp_drift = False))
				self.fail('Did not raise ValueError on subset_size > len(data)')
			except ValueError:
				pass

			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 1,
					avoid_fp_drift = False)), [1,2,3,4,5,6])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 2,
					avoid_fp_drift = False)),
					[1.5,2.5,3.5,4.5,5.5])
			self.assertEqual(list(movingaverage(map(float, [1,2,3,4,5,6]), 2,
					avoid_fp_drift = False)), [1.5,2.5,3.5,4.5,5.5])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 3,
					avoid_fp_drift = False)), [2,3,4,5])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 4,
					avoid_fp_drift = False)), [2.5,3.5,4.5])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 5,
					avoid_fp_drift = False)), [3,4])
			self.assertEqual(list(movingaverage([1,2,3,4,5,6], 6,
					avoid_fp_drift = False)), [3.5])

			self.assertEqual(list(movingaverage([40, 30, 50, 46, 39, 44],
					3, False, avoid_fp_drift = False)), [40.0,42.0,45.0,43.0])
			self.assertEqual(list(movingaverage([40, 30, 50, 46, 39, 44],
					3, True, avoid_fp_drift = False)), [40.0,42.0,45.0,43.0])


		#############################
		def test_XRangeRolling(self):
			try:
				list(movingaverage(range(1, 4), 0, avoid_fp_drift = False))
				self.fail('Did not raise ValueError on subset_size=0')
			except ValueError:
				pass

			try:
				list(movingaverage(range(1, 7), 7, avoid_fp_drift = False))
				self.fail('Did not raise ValueError on subset_size > len(data)')
			except ValueError:
				pass

			self.assertEqual(list(movingaverage(range(1, 7), 1,
					avoid_fp_drift = False)), [1,2,3,4,5,6])
			self.assertEqual(list(movingaverage(range(1, 7), 2,
					avoid_fp_drift = False)), [1.5,2.5,3.5,4.5,5.5])
			self.assertEqual(list(movingaverage(iter(map(float, range(1, 7))),
					2, avoid_fp_drift = False)), [1.5,2.5,3.5,4.5,5.5])
			self.assertEqual(list(movingaverage(range(1, 7), 3,
					avoid_fp_drift = False)), [2,3,4,5])
			self.assertEqual(list(movingaverage(range(1, 7), 4,
					avoid_fp_drift = False)), [2.5,3.5,4.5])
			self.assertEqual(list(movingaverage(range(1, 7), 5,
					avoid_fp_drift = False)), [3,4])
			self.assertEqual(list(movingaverage(range(1, 7), 6,
					avoid_fp_drift = False)), [3.5])


	######################################################################
	suite = unittest.TestLoader().loadTestsFromTestCase(TestMovingAverage)
	unittest.TextTestRunner(verbosity = 2).run(suite)
    
    