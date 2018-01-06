#pragma once
#include <string>

template<class T, int size>
class MinHeap {

public:
	MinHeap() : elementCount(0) 
	{}

	bool Insert(T element)
	{
		if(elementCount == size)
		{
			return false;
		}
		
		heap[elementCount] = element;
		upHeap(elementCount);
		elementCount++;
		return true;
	}

	bool Insert(T* elements, int n)
	{
		if(n > size-elementCount)
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

	T Pop()
	{
		if(elementCount < 1)
			throw "Heap is empty";

		T minElem = heap[0];

		elementCount--;
		heap[0] = heap[elementCount];
		downHeap(0);

		return minElem;
	}

	T Peek()
	{
		if(elementCount > 0)
			return heap[0];

		throw "Heap is empty";
	}

private:
	T heap[size];
	int elementCount;

	int getParentIndex(int childIndex)
	{
		if(childIndex == 0)
		{
			return -1;
		}
		
		return (childIndex - 1) / 2;
	}

	int getLeftChildIndex(int parentIndex)
	{
		return parentIndex * 2 + 1;
	}

	void upHeap(int index)
	{
		int parentIndex;
		T tmp;

		if(index != 0)
		{
			parentIndex = getParentIndex(index);

			if(heap[index] < heap[parentIndex])
			{
				tmp = heap[index];
				heap[index] = heap[parentIndex];
				heap[parentIndex] = tmp;
				upHeap(parentIndex);
			}
		}
	}

	void downHeap(int index)
	{
		int leftChildIndex = getLeftChildIndex(index);

		if(leftChildIndex >= elementCount)
		{
			return;
		}

		int childIndex;
		T tmp;

		if(leftChildIndex + 1 >= elementCount)
		{
			childIndex = leftChildIndex;
		}
		else
		{
			childIndex = heap[leftChildIndex] < heap[leftChildIndex + 1] ? leftChildIndex : leftChildIndex+1;
		}

		if(heap[childIndex] < heap[index])
		{
			tmp = heap[index];
			heap[index] = heap[childIndex];
			heap[childIndex] = tmp;
			downHeap(childIndex);
		}
	}
};
