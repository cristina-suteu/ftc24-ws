#include <stdio.h>
#include <stdlib.h>
#include <iio.h>

#define URI "local:"

int main() {

	struct iio_context *ctx;
	struct iio_device *dev;
	struct iio_channel *chn;
	char value[20];
	double temp_f; 
	double temp_c;

	ctx = iio_create_context_from_uri(URI);
	if(!ctx) {
		printf("No context\n");
		return -1;
	}
	dev = iio_context_find_device(ctx, "cpu_thermal");
	if(!dev) {
		printf("No such device \n");
		return -1;
	}
	chn = iio_device_find_channel(dev, "temp1", false);

	iio_channel_attr_read(chn, "input", value, sizeof(value));
	
	temp_c = atoi(value)/ 1000.0;
	temp_f = temp_c * 1.8 + 32 ;
	printf("%f Fahrenheit Degrees \n", temp_f);
	printf("%f Celsius Degrees \n", temp_c);

	iio_context_destroy(ctx);

	return 0;

}
