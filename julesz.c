#include <math.h>
#include <time.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

void getResponseRange(const double *filter, const int width, const int height, int *image, int* image2, int im_width, int im_height, double *min_response, double *max_response, int syn)
{
	int i, j, k, l;
	int width_index, height_index;
	double filtered1, filtered2;

	for (i = 0; i < im_height; i++)
		for (j = 0; j < im_width; j++)
		{
			filtered1 = 0;
			filtered2 = 0;
			for (k = 0; k < height; k++)
			{
				height_index = i + k;
				if (height_index < 0)
					height_index = height_index + im_height;
				else if (height_index >= im_height)
					height_index = height_index - im_height;

				for (l = 0; l < width; l++)
				{
					width_index = j + l;
					if (width_index < 0)
						width_index = width_index + im_width;
					else if (width_index >= im_width)
						width_index = width_index - im_width;

					filtered1 = filtered1 + filter[k*width+l] * image[height_index*im_width+width_index];
					filtered2 = filtered2 + filter[k*width+l] * image2[height_index*im_width+width_index];
				}
			}
			if (!syn)
			{
				if (filtered1 > max_response[0])
					max_response[0] = filtered1;
				if (filtered1 < min_response[0])
					min_response[0] = filtered1;
				if (filtered2 > max_response[0])
					max_response[0] = filtered2;
				if (filtered2 < min_response[0])
					min_response[0] = filtered2;
			}
		}
}

void getHistogram1(const double *filter, const int width, const int height, int *image, double *response, const int num_bins, double *filtered, int im_width, int im_height, double min_response, double max_response)
{
	int i, j, k, l;
	int index;
	int width_index, height_index;
	double mult = (double) (max_response - min_response) / num_bins;

	for (i = 0; i < im_height; i++)
		for (j = 0; j < im_width; j++)
		{
			index = i * im_width + j;
			filtered[index] = 0;
			for (k = 0; k < height; k++)
			{
				height_index = i + k;
				if (height_index < 0)
					height_index = height_index + im_height;
				else if (height_index >= im_height)
					height_index = height_index - im_height;

				for (l = 0; l < width; l++)
				{
					width_index = j + l;
					if (width_index < 0)
						width_index = width_index + im_width;
					else if (width_index >= im_width)
						width_index = width_index - im_width;

					filtered[index] = filtered[index] + filter[k*width+l] * image[height_index*im_width+width_index];
				}
			}
		}

	for (i = 0; i < num_bins; i++)
		response[i] = 0;

	for (i = 0; i < im_width * im_height; i++)
	{	
		if (filtered[i] >= min_response && filtered[i] <= max_response)
		{
			index = floor((filtered[i] - min_response) / mult);
			if (index >= 0 && index < num_bins)
				response[index] = response[index] + 1;
		}
	}
}

void getHistogram2(const double **filters, const int *width, const int *height, const int num_filters, int *image, int *synthesized, const int im_width, const int im_height, int num_bins, double *response, double *min_response, double *max_response, int syn)
{
	int i;

	double *filtered;
	filtered = (double *)malloc(im_width * im_height * sizeof(double));

	for (i = 0; i < num_bins * num_filters; i++)
		response[i] = 0;

	for (i = 0; i < num_filters; i++)
	{	
		if (!syn)
			getResponseRange(filters[i], width[i], height[i], image, synthesized, im_width, im_height, min_response, max_response, syn);
		getHistogram1(filters[i], width[i], height[i], image, &(response[i * num_bins]), num_bins, filtered, im_width, im_height, min_response[0], max_response[0]);
	}
	
	free(filtered);
}

void Gibbs(const double *weights, const double **filters, const int *width, const int *height, const int num_filters, const int num_bins, int *orig_img, int *synthesized, double *final_response, int im_width, int im_height, int max_intensity)
{
	time_t start;
	time_t end;

	int i, j, k, l, x, y, x1, x2, y1, y2, W, H, gray, index1, index2, minimum, num_iter;

	double T;
	double mult, random_num;
	double sum_probs, sum_error;
	double *min_response;
	double *max_response;

	double *syn_response;
	double **orig_response;
	double **diff_response;
	double **syn_filtered;
	double **orig_filtered;
	double *difference;
	double *probs;

	orig_response = (double **)malloc(num_filters * sizeof(double *));
	syn_response = (double *)malloc(num_bins * sizeof(double));
	diff_response = (double **)malloc(num_filters * sizeof(double *));
	syn_filtered = (double **)malloc(num_filters * sizeof(double *));
	orig_filtered = (double **)malloc(num_filters * sizeof(double *));
	difference = (double *)malloc(num_bins * sizeof(int));
	probs = (double *)malloc((max_intensity + 1) * sizeof(double));
	min_response = (double *)malloc(num_filters * sizeof(double));
	max_response = (double *)malloc(num_filters * sizeof(double));

	for (i = 0; i < num_filters; i++)
	{
		min_response[i] = INFINITY;
		max_response[i] = -INFINITY;
		getResponseRange(filters[i], width[i], height[i], orig_img, synthesized, im_width, im_height, &(min_response[i]), &(max_response[i]), 0);
		syn_filtered[i] = (double *)malloc(im_width * im_height * sizeof(double));
		orig_filtered[i] = (double *)malloc(im_width * im_height * sizeof(double));
		diff_response[i] = (double *)malloc(num_bins * sizeof(double));
		orig_response[i] = (double *)malloc(num_bins * sizeof(double));
		getHistogram1(filters[i], width[i], height[i], orig_img, orig_response[i], num_bins, orig_filtered[i], im_width, im_height, min_response[i], max_response[i]);
		getHistogram1(filters[i], width[i], height[i], synthesized, syn_response, num_bins, syn_filtered[i], im_width, im_height, min_response[i], max_response[i]);

		for (j = 0; j < num_bins; j++){
			diff_response[i][j] = orig_response[i][j] - syn_response[j];
		}
	}

	T = 0.1;
	num_iter = 0;
	sum_error = INT_MAX;

	while (sum_error > 0 && num_iter < 50) // annealing
	{
		time(&start);
		for (i = 0; i < im_height; i++)
		{
			for (j = 0; j < im_width; j++)
			{
				gray = synthesized[i*im_width+j];

				for (k = 0; k <= max_intensity; k++)
					probs[k] = 0;

				// for every possible intensity value, 0-255.
				for (k = -gray; k <= max_intensity-gray; k++)
				{
					//for every filter
					for (l = 0; l < num_filters; l++)
					{
						W = width[l];
						H = height[l];
						mult = (double)(max_response[l] - min_response[l]) / num_bins;

						for (x = 0; x < num_bins; x++)
							difference[x] = diff_response[l][x];

						for (x = 0; x < H; x++)
						{
							x1 = H - x - 1;
							x2 = i - x1;
							if (x2 < 0)
								x2 = x2 + im_height;
							else if (x2 >= im_height)
								x2 = x2 - im_height;

							for (y = 0; y < W; y++)
							{
								y1 = W - y - 1;
								y2 = j - y1;
								if (y2 < 0)
									y2 = y2 + im_width;
								else if (y2 >= im_width)
									y2 = y2 - im_width;

								//change the histograms
								index1 = floor((syn_filtered[l][x2*im_width+y2] - min_response[l]) / mult);
								index2 = floor((syn_filtered[l][x2*im_width+y2] + k * filters[l][x1*W+y1] - min_response[l]) / mult);

								if (index1 != index2 && index1 >= 0 && index1 < num_bins && index2 >= 0 && index2 < num_bins)
								{
									difference[(int)index1] = difference[(int)index1] + 1;
									difference[(int)index2] = difference[(int)index2] - 1;
								}
							}
						}

						for (x = 0; x < num_bins; x++)
							probs[k+gray] = probs[k+gray] + fabs(difference[x]) * weights[x];
					}
				}

				minimum = INT_MAX;
				for (k = 0; k <= max_intensity; k++)
					if (probs[k] < minimum)
						minimum = probs[k];

				sum_probs = 0;
				for (k = 0; k <= max_intensity; k++)
				{
					probs[k] = probs[k] - minimum;
					sum_probs = sum_probs + probs[k];
				}

				for (k = 0; k <= max_intensity; k++)
				{
					probs[k] = probs[k] / sum_probs;
					probs[k] = exp(-probs[k] / T);
				}

				sum_probs = 0;
				for (k = 0; k <= max_intensity; k++)
					sum_probs = sum_probs + probs[k];
				for (k = 0; k <= max_intensity; k++)
					probs[k] = probs[k] / sum_probs;

				random_num = (double)rand() / RAND_MAX;
				for (k = 0; k <= max_intensity; k++)
				{
					if (random_num < probs[k])
					{
						synthesized[i*im_width+j] = k;
						break;
					}
					else
						random_num = random_num - probs[k];
				}

				for (l = 0; l < num_filters; l++)
				{
					W = width[l];
					H = height[l];
					mult = (double)(max_response[l] - min_response[l]) / num_bins;
					for (x = 0; x < H; x++)
					{
						x1 = H - x - 1;
						x2 = i - x1;
						if (x2 < 0)
							x2 = x2 + im_height;
						else if (x2 >= im_height)
							x2 = x2 - im_height;

						for (y = 0; y < W; y++)
						{
							y1 = W - y - 1;
							y2 = j - y1;
							if (y2 < 0)
								y2 = y2 + im_width;
							else if (y2 >= im_width)
								y2 = y2 - im_width;

							//change the histograms
							index1 = (int)floor((syn_filtered[l][x2*im_width+y2] - min_response[l]) / mult);
							index2 = (int)floor((syn_filtered[l][x2*im_width+y2] + (synthesized[i*im_width+j] - gray) * filters[l][x1*W+y1] - min_response[l]) / mult);
							
							if (index1 >= 0 && index1 < num_bins && index2 >= 0 && index2 < num_bins)
							{
								diff_response[l][index1] = diff_response[l][index1] + 1;
								syn_filtered[l][x2*im_width+y2] = syn_filtered[l][x2*im_width+y2] + (synthesized[i*im_width+j] - gray) * filters[l][x1*W+y1];
								diff_response[l][index2] = diff_response[l][index2] - 1;
							}
							
						}
					}
				}
			}
		}
		T = T * 0.96;
		num_iter = num_iter + 1;

		sum_error = 0;
		for (i = 0; i < num_filters; i++)
			for (j = 0; j < num_bins; j++)
				sum_error = sum_error + fabs(diff_response[i][j]);

		sum_error = sum_error / num_filters / num_bins;

		time(&end);
		printf("Iteration %d took %.2lf minutes, error = %.6lf\n", num_iter, difftime(end, start)/60, sum_error);
	}

	for (i = 0; i < num_filters; i++)
		for (j = 0; j < num_bins; j++)
			final_response[j*num_filters+i] = (orig_response[i][j] - diff_response[i][j]) / im_width / im_height;

	for (i = 0; i < num_filters; i++)
	{
		free(diff_response[i]);
		free(orig_response[i]);
		free(syn_filtered[i]);
		free(orig_filtered[i]);
	}

	free(syn_response);
	free(diff_response);
	free(syn_filtered);
	free(orig_response);
	free(orig_filtered);
	free(difference);
	free(min_response);
	free(max_response);
	free(probs);
}