#pragma once
#include <string>
#include <crt/host_defines.h>
#include <driver_types.h>

//actually not usable on cuda device because of recursion :(
template<class T, int size>
class MinHeap {

public:
	__host__ __device__
	MinHeap() : elementCount(0)
	{}

	__host__ __device__
	bool Insert(T element)
	{
		if (elementCount == size)
		{
			return false;
		}

		heap[elementCount] = element;
		upHeap(elementCount);
		elementCount++;
		return true;
	}

	__host__ __device__
	bool Insert(T* elements, int n)
	{
		if (n > size - elementCount)
		{
			return false;
		}

		bool success = true;

		for (int i = 0; i < n; ++i)
		{
			success &= Insert(elements[i]);
		}
		return success;
	}
	
	// remove and return root element
	__host__ __device__
	T Pop()
	{
		if (elementCount < 1)
		{
#ifdef __CUDA_ARCH__
			return T();
#else
			throw "Heap is empty";

#endif	
		}

		T minElem = heap[0];

		elementCount--;
		heap[0] = heap[elementCount];
		downHeap(0);

		return minElem;
	}
	
	// return root element
	__host__ __device__
	T Peek()
	{
		if (elementCount > 0)
			return heap[0];

#ifdef __CUDA_ARCH__
		return T();
#else
		throw "Heap is empty";

#endif	
	}

private:
	T heap[size];
	int elementCount;

	__host__ __device__
	int getParentIndex(int childIndex)
	{
		if (childIndex == 0)
		{
			return -1;
		}

		return (childIndex - 1) / 2;
	}

	__host__ __device__
	int getLeftChildIndex(int parentIndex)
	{
		return parentIndex * 2 + 1;
	}

	// let new element float up to find position in heap
	__host__ __device__
	void upHeap(int index)
	{
		int parentIndex;
		T tmp;

		if (index > 0)
		{
			parentIndex = getParentIndex(index);

			if (heap[index] < heap[parentIndex])
			{
				tmp = heap[index];
				heap[index] = heap[parentIndex];
				heap[parentIndex] = tmp;
				upHeap(parentIndex);
			}
		}
	}

	// let element sink down to find right position
	__host__ __device__
	void downHeap(int index)
	{
		int leftChildIndex = getLeftChildIndex(index);

		if (leftChildIndex >= elementCount)
		{
			return;
		}

		int childIndex;
		T tmp;

		if (leftChildIndex + 1 >= elementCount)
		{
			childIndex = leftChildIndex;
		}
		else
		{
			childIndex = heap[leftChildIndex] < heap[leftChildIndex + 1] ? leftChildIndex : leftChildIndex + 1;
		}

		if (heap[childIndex] < heap[index])
		{
			tmp = heap[index];
			heap[index] = heap[childIndex];
			heap[childIndex] = tmp;
			downHeap(childIndex);
		}
	}
};
