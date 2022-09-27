function param = setProxIFParameters(param,N)
% Proximity Isolation Forest, v1.0, 2022
% (c) A. Mensi
%
% param = setProxIFParameters(param,N) checks and sets the parameters of
% the ProxIF model to be built. 
% param is the structure of parameters
% N is the number of training objects
if nargin<2
    error("You should pass the number of objects in the training set.")
end
if ~isfield(param,'T') || param.T<1 
    param.T=500;
elseif ~isequal(class(param.T),'double')
    error("T should be a number!");
end

if ~isfield(param,'S')
    param.S=min(256,N);
elseif ~isequal(class(param.S),'double')
    error("S should be a number!");
end

if param.S>N
    param.S=N;
end

if ~isfield(param,'D')
    param.D='log';
elseif ~any(strcmpi(param.D,{'log','max'}))
    error("D should either be 'log' or 'max'");
end
if equal(param.D,'log') %Needed to make the parameter setting more user friendly + if S is wrongly set to S>N
    param.max_depth=ceil(log2(param.S));
else %param.D is 'max'
    param.max_depth=param.S-1;
end


if ~isfield(param,'crit')
    param.crit='O-2PH';
else
    if ~any(strcmpi(param.crit,{'R-1P', 'R-2P', 'O-1PSD', 'O-2PSD', 'O-2PSP', 'O-1PH', 'O-2PH','O-1PRD', 'O-2PRD'}))
        error("You should choose a training strategy among 'R-1P', 'R-2P', 'O-1PSD', 'O-2PSD', 'O-2PSP', 'O-1PH', 'O-2PH','O-1PRD', 'O-2PRD'");
    end
end

if any(strcmpi(param.crit,{'O-1PRD','O-2PRD'}))
    if ~isfield(param,'alpha')
        param.alpha=0.9999;
    elseif ~isequal(class(param.alpha),'double')
        error("alpha should be a number!");
    end
end


if any(strcmpi(param.crit,{'R-1P','R-2P'}))
    param.max_comp=1;
else
    if ~isfield(param,'max_comp')
        param.max_comp=20;
    elseif ~isequal(class(param.max_comp),'double')
        error("R should be a number!");
    end

end

if contains(param.crit,'1P') %Needed to better handle the tree training
    param.thr=true;
else
    param.thr=false;
end

end