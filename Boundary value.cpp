#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#define grid_N 500

using namespace std;

class Data
{
	double left_;
	double right_;
	double bottom_;
	double top_;

public:
	Data(double left, double right, double bottom, double top) :
		left_(left),
		right_(right),
		bottom_(bottom),
		top_(top)
	{};

	double u(double x, double y)
	{
		return exp(1 - (x + y)*(x + y));
	}

	double k(double x, double y)
	{
		return 4 + x;
	}
	double q(double x, double y)
	{
		double result = x + y;
		if (result > 0) return result * result;
		else return 0;
	}

	double fi(double x, double y)
	{
		return exp(1 - (x + y)*(x + y));
	}

	double F(double x, double y)
	{
		double ddx = 2 * exp(1 - (x + y)*(x + y)) * (2 * x*x*x + 4 * x*x*(y + 2) + 2 * x*(y*y + 8 * y - 1) + 8 * y*y - y - 4);
		double ddy = 2 * (x + 4) * exp(1 - (x + y)*(x + y)) * (2 * x*x + 4 * x * y + 2 * y*y - 1);
		return -(ddx + ddy) + q(x, y) * u(x, y);
		//return -2 * exp(1 - (x + y)*(x + y))*(4 * (x + 4)*y*y + 8 * x*(x + 4)*y + x * (4 * x*(x + 4) - 3) - y - 8) + q(x, y) * u(x, y);
	}

	double getl() { return left_; }
	double getr() { return right_; }
	double getb() { return bottom_; }
	double gett() { return top_; }
};

class Dif_Method
{
	int M_;
	int N_;
	int proc_;
	int size;
	double h1;
	double h2;
	double eps;
	Data &data;
	vector <double> B;
	vector <double> w_vector;
	vector <double> r;
	vector <double> Ar;
	vector <double> diff;
	vector <int> recvcounts;
	vector <int> displs;

public:
	Dif_Method(int M, int N, Data &data, int proc) :
		M_(M),
		N_(N),
		data(data),
		proc_(proc),
		size((M - 1) * (N - 1)),
		w_vector(size, 0.0),
		B(size, 0.0),
		r(size, 0.0),
		Ar(size, 0.0),
		diff(size, 0.0),
		recvcounts(proc_ + 1, 0),
		displs(proc_ + 1, 0)
	{
		h1 = (data.getr() - data.getl()) / M_;
		h2 = (data.gett() - data.getb()) / N_;
		eps = 0.00001;
		Init();
	}

	void Init()
	{
		vector <double> A(5, 0.0);
		MPI_Bcast(&M_, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&N_, 1, MPI_INT, 0, MPI_COMM_WORLD);

		int height = 0;
		int rows = size / proc_;
		int bonus = size % proc_;
		int offset = 0;

		recvcounts[0] = 0;
		displs[0] = 0;
		for (int i = 1; i <= proc_; i++) {
			if (i <= bonus) height = rows + 1;
			else height = rows;
			recvcounts[i] = height;
			displs[i] = offset;
			offset += height;
		}
		MPI_Bcast(recvcounts.data(), proc_ + 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(displs.data(), proc_ + 1, MPI_INT, 0, MPI_COMM_WORLD);

		int cur_proc = 1;                                 //number of the current proc
		int counter_rows = recvcounts[cur_proc];          //q of rows left to be pushed
		for (int j = 1; j <= N_ - 1; j++) {
			for (int i = 1; i <= M_ - 1; i++) {
				int number = (i - 1) + (j - 1) * (M_ - 1);  //number of w_ij in vector 
				double xi = data.getl() + i * h1;
				double yj = data.getb() + j * h2;
				double kx1 = data.k(xi - h1 / 2, yj) / (h1*h1);  //w_(i-1)j
				double kx2 = data.k(xi + h1 / 2, yj) / (h1*h1);  //w_(i+1)j
				double ky1 = data.k(xi, yj - h2 / 2) / (h2*h2);  //w_i(j-1)
				double ky2 = data.k(xi, yj - h2 / 2) / (h2*h2);  //w_i(j+1)
				A[2] = kx1 + kx2 + ky1 + ky2 + data.q(xi, yj);
				if (i > 1 && j > 1 && i < M_ - 1 && j < N_ - 1) {
					A[0] = -kx1;
					A[1] = -ky1;
					A[3] = -ky2;
					A[4] = -kx2;
					B[number] = data.F(xi, yj);
				}
				else if (j == 1 && i > 1 && i < M_ - 1) {
					A[0] = -kx1;
					A[1] = 0.0;
					A[3] = -ky2;
					A[4] = -kx2;
					B[number] = data.F(xi, yj) + ky1 * data.fi(xi, data.getb());
				}
				else if (j == N_ - 1 && i > 1 && i < M_ - 1) {
					A[0] = -kx1;
					A[1] = -ky1;
					A[3] = 0.0;
					A[4] = -kx2;
					B[number] = data.F(xi, yj) + ky2 * data.fi(xi, data.gett());
				}
				else if (i == 1 && j > 1 && j < N_ - 1) {
					A[0] = 0.0;
					A[1] = -ky1;
					A[3] = -ky2;
					A[4] = -kx2;
					B[number] = data.F(xi, yj) + kx1 * data.fi(data.getl(), yj);
				}
				else if (i == M_ - 1 && j > 1 && j < N_ - 1) {
					A[0] = -kx1;
					A[1] = -ky1;
					A[3] = -ky2;
					A[4] = 0.0;
					B[number] = data.F(xi, yj) + kx2 * data.fi(data.getr(), yj);
				}
				else if (i == 1 && j == 1) {
					A[0] = 0.0;
					A[1] = 0.0;
					A[3] = -ky2;
					A[4] = -kx2;
					B[number] = data.F(xi, yj) + kx1 * data.fi(data.getl(), yj) + ky1 * data.fi(xi, data.getb());
				}
				else if (i == M_ - 1 && j == 1) {
					A[0] = -kx1;
					A[1] = 0.0;
					A[3] = -ky2;
					A[4] = 0.0;
					B[number] = data.F(xi, yj) + kx2 * data.fi(data.getr(), yj) + ky1 * data.fi(xi, data.getb());
				}
				else if (i == 1 && j == N_ - 1) {
					A[0] = 0.0;
					A[1] = -ky1;
					A[3] = 0.0;
					A[4] = -kx2;
					B[number] = data.F(xi, yj) + kx1 * data.fi(data.getl(), yj) + ky2 * data.fi(xi, data.gett());
				}
				else if (i == M_ - 1 && j == N_ - 1) {
					A[0] = -kx1;
					A[1] = -ky1;
					A[3] = 0.0;
					A[4] = 0.0;
					B[number] = data.F(xi, yj) + kx2 * data.fi(data.getr(), yj) + ky2 * data.fi(xi, data.gett());
				}
				MPI_Send(A.data(), 5, MPI_DOUBLE, cur_proc, 1, MPI_COMM_WORLD);   //send 1 row of A
				counter_rows--;
				if (counter_rows == 0) {
					MPI_Send(&B[number - recvcounts[cur_proc] + 1], recvcounts[cur_proc], MPI_DOUBLE, cur_proc, 1, MPI_COMM_WORLD); //send proc's part of B
					cur_proc++;
					if (cur_proc <= proc_) counter_rows = recvcounts[cur_proc];
				}
			}
		}
	}

	void Solve(const char *filename_approx, const char *filename_exact)
	{
		double difference = eps + 1.0;
		double tau = 0.0;
		vector <double> null(3, 0.0);
		vector <double> res(3, 0.0);
		//solving
		while (difference - eps > 0) {
			MPI_Allgatherv(w_vector.data(), 0, MPI_DOUBLE, r.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
			MPI_Allreduce(null.data(), res.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //[r,r], [Ar, Ar], [Ar, r]
			double tau = res[2] / res[1];
			difference = sqrt(res[0] * h1 * h2) * tau;
			MPI_Bcast(&difference, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Allgatherv(null.data(), 0, MPI_DOUBLE, w_vector.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
		}

		/*//printing result
		fstream file1(filename_approx, ios::out | ios::trunc);
		fstream file2(filename_exact, ios::out | ios::trunc);
		for (int i = 0; i < size; i++) {
			double xi = data.getl() + (i % (M_ - 1) + 1) * h1;
			double yi = data.getb() + (i / (M_ - 1) + 1) * h2;
			file1 << xi << " " << yi << " " << w_vector[i] << endl;
			file2 << xi << " " << yi << " " << data.u(xi, yi) << endl;
		}
		file1.close();
		file2.close();*/
	}
};

class Slave
{
	vector <int> recvcounts_;
	vector <int> displs_;
	int rank_;
	int M_;
	int N_;
	int height;
	int length;
	double eps;
	const vector <vector <double> > A;
	const vector <double> B;
	MPI_Status stat;

public:
	Slave(int r, int M, int N, vector <int> recvc, vector <int> dsp, vector <vector <double> > op, vector <double> sub) :
		rank_(r),
		M_(M),
		N_(N),
		recvcounts_(recvc),
		displs_(dsp),
		A(op),
		B(sub),
		eps(0.00001)
	{
		height = recvcounts_[rank_];
		length = (M_ - 1) * (N_ - 1);
	}

	void Mult(vector <double> &x, bool sub, vector <double> &result)
	{
		int row_len = M_ - 1;
		int cur_num = displs_[rank_];
		double res = 0.0;
		//#pragma omp parallel for
		for (int i = 0; i < height; i++) {
			res = A[i][2] * x[cur_num + i];
			if (A[i][0] != 0.0) res += A[i][0] * x[cur_num + i - 1];
			if (A[i][1] != 0.0) res += A[i][1] * x[cur_num + i - row_len];
			if (A[i][3] != 0.0) res += A[i][3] * x[cur_num + i + row_len];
			if (A[i][4] != 0.0) res += A[i][4] * x[cur_num + i + 1];
			if (sub) result[i] = res - B[i];
			else result[i] = res;
		}
	}

	vector <double> Scalar(vector <double> &r, vector <double> &Ar)
	{
		vector <double> result(3, 0.0);
		int start_number = displs_[rank_];
		int row_len = M_ - 1;
		//#pragma omp parallel for
		for (int n = 0; n < height; n++) {
			int i = (start_number + n) % row_len;
			int j = (start_number + n) / row_len;
			double p = 1.0;
			if (i == 0 || i == M_ - 2) p *= 0.5;
			if (j == 0 || j == N_ - 2) p *= 0.5;
			result[0] += r[n] * r[n] * p; //rr += r[n] * r[n] * p;
			result[1] += Ar[n] * Ar[n] * p; //ArAr += Ar[n] * Ar[n] * p;
			result[2] += Ar[n] * r[n] * p; //Arr += Ar[n] * r[n] * p
		}
		return result;
	}

	void work()
	{
		double dif = eps + 1.0;
		vector <double> all_w(length, 0.0);
		vector <double> all_r(length, 0.0);
		vector <double> my_w(height, 0.0);
		vector <double> my_r(height, 0.0);
		vector <double> my_ar(height, 0.0);
		vector <double> scal(3, 0.0);
		vector <double> res_scal(3, 0.0);
		while (dif - eps > 0) {
			Mult(all_w, true, my_r);                                           //compute r
			MPI_Allgatherv(my_r.data(), height, MPI_DOUBLE, all_r.data(), recvcounts_.data(), displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
			Mult(all_r, false, my_ar);                                         //compute Ar

			scal = Scalar(my_r, my_ar);
			MPI_Allreduce(scal.data(), res_scal.data(), 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); //[r,r], [Ar, Ar], [Ar, r]
			double tau = res_scal[2] / res_scal[1];
			for (int i = 0; i < height; i++)
				my_w[i] += -(tau * my_r[i]);
			MPI_Bcast(&dif, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			MPI_Allgatherv(my_w.data(), height, MPI_DOUBLE, all_w.data(), recvcounts_.data(), displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
		}
	}
};

int main(int argc, char **argv)
{
	int rank, size;
	if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
		printf("Failed to initialize MPI.");
		return -1;
	}
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//cout << rank << endl;
	if (rank == 0) {     //master branch
		double time_started = MPI_Wtime();
		Data data(-1, 2, -2, 2);
		Dif_Method dif(grid_N, grid_N, data, size - 1);
		dif.Solve("out1.txt", "out2.txt");
		double time_finish = MPI_Wtime() - time_started;
		cout << time_finish << endl;
	}
	else {               //slave branch
		MPI_Status st;
		vector <int> recvcounts(size, 0);
		vector <int> displs(size, 0);
		int M = 0;
		int N = 0;
		MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(recvcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(displs.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

		vector <vector <double> > A(recvcounts[rank], vector <double>(5, 0.0));
		vector <double> B(recvcounts[rank], 0.0);
		for (int i = 0; i < recvcounts[rank]; i++) {
			MPI_Recv(A[i].data(), 5, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
		}
		MPI_Recv(B.data(), recvcounts[rank], MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
		Slave slave(rank, M, N, recvcounts, displs, A, B);
		slave.work();
	}
	MPI_Finalize();

	return 0;
}
