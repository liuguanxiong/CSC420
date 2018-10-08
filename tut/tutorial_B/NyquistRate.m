%% Fourier transform of a sin wave + noise

fs = 100;               % sampling frequency
t = 0:(1/fs):(10-1/fs); % time vector
s = cos(2*pi*15*t) + 3 * randn(size(t));
subplot(3,1,1); plot(s)
n = length(s);
S = fft(s);
f = (0:n-1)*(fs/n);     %frequency range
power = abs(S).^2/n;    %power
subplot(3,1,2); plot(f,power)

Y = fftshift(S);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(3,1,3); plot(fshift,powershift)

%% let's make a singal (x) where most of the energy is between 0.1 to 5 Hz

x = 5 * rand(size(t));
for f = 0.5 : 0.001 : 5
    x = x + rand * cos(2*pi*f*t + rand*2*pi);
end
x = x - mean(x(:));

subplot(2,1,1); plot(x)
n = length(x);
X = fft(x);

f = (0:n-1)*(fs/n);     %frequency range
power = abs(X).^2/n;    %power

Y = fftshift(X);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(2,1,2); plot(fshift,powershift)


%% Let's multiply these two signal

s = cos(2*pi*15*t); % but let's use a clean sine wave instead of a noisy one

xs = x .* s;

subplot(3,1,1); plot(x)
subplot(3,1,2); plot(xs)
n = length(xs);
XS = fft(xs);

f = (0:n-1)*(fs/n);     %frequency range
power = abs(XS).^2/n;    %power

Y = fftshift(XS);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(3,1,3); plot(fshift,powershift)


%% Fourier Transform of "sampling" (with high frequency)

fs = 100;               % sampling frequency
t = 0:(1/fs):(10-1/fs); % time vector
s1 = zeros(size(t));

s1( 1 : 8 : end) = 1;
subplot(2,1,1); plot(s1)
n = length(s1);
S1 = fft(s1);
f = (0:n-1)*(fs/n);     %frequency range
power = abs(S1).^2/n;    %power

Y = fftshift(S1);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(2,1,2); plot(fshift,powershift)

%% Fourier Transform of "sampling" (with medium frequency)

fs = 100;               % sampling frequency
t = 0:(1/fs):(10-1/fs); % time vector
s2 = zeros(size(t));

s2( 1 : 25 : end) = 1;
subplot(2,1,1); plot(s2)
n = length(s2);
S2 = fft(s2);
f = (0:n-1)*(fs/n);     %frequency range
power = abs(S2).^2/n;    %power

Y = fftshift(S2);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(2,1,2); plot(fshift,powershift)

%% Fourier Transform of "sampling" (with low frequency)

fs = 100;               % sampling frequency
t = 0:(1/fs):(10-1/fs); % time vector
s3 = zeros(size(t));

s3( 1 : 100 : end) = 1;
subplot(2,1,1); plot(s3)
n = length(s3);
S3 = fft(s3);
f = (0:n-1)*(fs/n);     %frequency range
power = abs(S3).^2/n;    %power

Y = fftshift(S3);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(2,1,2); plot(fshift,powershift)

%% Let's sample our signal with high frequency

x2 = x .* s1;

subplot(3,1,1); plot(x)

subplot(3,1,2); plot(x2)
n = length(x2);
X1 = fft(x2);

f = (0:n-1)*(fs/n);     %frequency range
power = abs(X1).^2/n;    %power

Y = fftshift(X1);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(3,1,3); plot(fshift,powershift)

%% Let's sample our signal with medium frequency

x2 = x .* s2;

subplot(3,1,1); plot(x)
subplot(3,1,2); plot(x2)
n = length(x2);
X2 = fft(x2);

f = (0:n-1)*(fs/n);     %frequency range
power = abs(X2).^2/n;    %power

Y = fftshift(X2);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(3,1,3); plot(fshift,powershift)

%% Let's sample our signal with low frequency

x3 = x .* s3;

subplot(3,1,1); plot(x)
subplot(3,1,2); plot(x3)
n = length(x3);
X3 = fft(x3);

f = (0:n-1)*(fs/n);     %frequency range
power = abs(X3).^2/n;    %power

Y = fftshift(X3);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(3,1,3); plot(fshift,powershift)

%% 

x = rand(size(t))/20;
x = x - mean(x(:));
x = x + exp(5 * -t); % 5 % try different numbers (1, 2, 20, 50)


subplot(2,1,1); plot(x)
n = length(x);
X = fft(x);

f = (0:n-1)*(fs/n);     %frequency range
power = abs(X).^2/n;    %power

Y = fftshift(X);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(2,1,2); plot(fshift,powershift)

%% Let's sample our signal with high frequency

x2 = x .* s1;

subplot(3,1,1); plot(x)
subplot(3,1,2); plot(x2)
n = length(x2);
X1 = fft(x2);

f = (0:n-1)*(fs/n);     %frequency range
power = abs(X1).^2/n;    %power

Y = fftshift(X1);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(3,1,3); plot(fshift,powershift)

%% Let's sample our signal with medium frequency

x2 = x .* s2;

subplot(3,1,1); plot(x)
subplot(3,1,2); plot(x2)
n = length(x2);
X2 = fft(x2);

f = (0:n-1)*(fs/n);     %frequency range
power = abs(X2).^2/n;    %power

Y = fftshift(X2);
fshift = (-n/2:n/2-1)*(fs/n); % zero-centered frequency range
powershift = abs(Y).^2/n;     % zero-centered power
subplot(3,1,3); plot(fshift,powershift)

