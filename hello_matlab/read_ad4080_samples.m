rx = adi.AD4080.Rx;
rx.uri = 'ip:172.18.228.187'; % Change this to the IP address on which iiod is listening

rx.SamplesPerFrame = 16384;
rx.EnabledChannels = [1];

data = rx();

figure(1);
plot(data);

% Delete the system object
release(rx);
