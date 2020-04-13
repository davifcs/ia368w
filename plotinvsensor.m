function plotinvsensor()

  % range maximo
  Rmax = 10;
  % abertura maxima |A|
  Amax = 5.0;
  % leitura do sensor
  d = 5;
  mu = [d 0];

  sigma = [0.5 0 ; 0 0.05];   % altera o shape da curva
%  sigma = [0.05 0 ; 0 0.005];   % altera o shape da curva
% sigma = [0.005 0 ; 0 0.0005];   % altera o shape da curva

  invSigma = inv(sigma);
  Pmin = 0.1;
  K = 0.02;   % altera o maximo da curva
%  K = 0.0002;   % altera o maximo da curva
%  K = 0.000004;   % altera o maximo da curva

  x = linspace(0, Rmax, 400);  % distancia
  y = linspace(-Amax, Amax, 400);   % angulo
  [xx, yy] = meshgrid(x, y);
  max = 0;

  for i=1:length(x)
       for j=1:length(y)
            if x(i) < mu(1)
               P = Pmin;
            else
               P = 0.5;
            end
            X = [x(i) y(j)];
            % OBS: no eixo x sao plotadas as COLUNAS de z
            z(j,i) = P + (K/(2*pi*sigma(1,1)*sigma(2,2)) + 0.5 - P)*exp(-.5 * (X - mu) * invSigma * (X - mu)');
            if z(j,i) > max
               max = z(j,i);
            end
       end
  end
  max

  surf(xx,yy,z);
  xlabel('distancia')
  ylabel('angulo')
  zlabel('p(x|z,theta)')
  colormap(hsv)
end

