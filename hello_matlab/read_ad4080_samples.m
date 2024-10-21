rx = adi.AD4080.Rx;
rx.uri = 'ip:10.48.65.170:50906'; % Change this to the IP address on
                                  % which iiod is listening
rx.SamplesPerFrame = 16384;
rx.EnabledChannels = [1];

data = rx();
data = data * rx.Scale / 1e3; % Scale is in mV

plot(data);

% Delete the system object
release(rx);
