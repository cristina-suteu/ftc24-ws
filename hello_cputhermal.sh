
URI="local:"

while true
do
temp_mc=$(iio_attr -u $URI -c cpu_thermal temp1 input)
temp_c=$(echo "$temp_mc/1000" | bc)
temp_f=$(echo "($temp_mc/1000)*1.8 + 32" | bc)
echo "CPU Temperature is $temp_f Fahrenheit Degress"
echo "And $temp_c Celsius Degrees"
sleep 5
done
