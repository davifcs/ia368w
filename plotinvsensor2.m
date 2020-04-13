function plotinvsensor2()

  % tamanho da celula
  tamCel = 50;     % mm
  % numero de celulas
  numCelX = 200;     % 10 x 10 m
  numCelY = 200;
  % mapa
  mapa = zeros(numCelX, numCelY);

  % pose do robo
  px = 3000;
  py = 1500;
  pth = 15*pi/180;

  % leitura do laser
  range = 2000;
  fi = 20*pi/180;
  mu = [range fi];

  % dados do laser
  precisao = 50;  % mm
  passo = 1*pi/180;   % rad

  sigma = [2 0 ; 0 0.2];   % altera o shape da curva
% sigma = [0.5 0 ; 0 0.05];   % altera o shape da curva
% sigma = [0.05 0 ; 0 0.005];   % altera o shape da curva
  invSigma = inv(sigma);
  Pmin = 0.4;

  K = 0.5;   % altera o maximo da curva (0.7-0.8)
%  K = 0.02;   % altera o maximo da curva (0.7-0.8)
%  K = 0.0002;   % altera o maximo da curva

  x = linspace(1, numCelX, numCelX);  % X
  y = linspace(1, numCelY, numCelY);  % Y
  [xx, yy] = meshgrid(x, y);
  max = 0;

  z = zeros(length(y), length(x));
  for i=1:length(x)
    % posicao da celula (i,j) no referencial global
    xi = (i-1)*tamCel + tamCel/2;
    for j=1:length(y)
       yj = (j-1)*tamCel + tamCel/2;
       % raio em relacao ao robo
       r = sqrt((xi-px)^2 + (yj-py)^2);
       % angulo em relacao ao robo
       b = atan2((yj-py), (xi-px)) - pth;
       % diferença em relacao a leitura do sensor
       delta = [r b] - mu;
       if abs(delta(2)) > 2*passo
%       if abs(delta(2)) > 2*passo || abs(delta(1)) > 2*precisao
         continue;
       end
       if r < mu(1)
          P = Pmin;
       else
          P = 0.5;
       end
       delta(1) = delta(1)/1000;   % distancia em metros
       z(i,j) = P + (K/(2*pi*sigma(1,1)*sigma(2,2)) + 0.5-P)*exp(-0.5*delta*invSigma*delta');
       mapa(i,j) = mapa(i,j) + log(z(i,j)/(1-z(i,j)));
       if z(i,j) > max
          max = z(i,j);
       end
     end
  end
  max

  surf(xx,yy,z');
  xlabel('X')
  ylabel('Y')
  zlabel('p(x|z,theta)')
  colormap(hsv)

end

