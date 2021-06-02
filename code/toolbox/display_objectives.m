function display_objectives(OBJECTIVES, NAMES, TIMES, name,format)

if nargin <4
    name = ''; 
end
if nargin <5
    format = 'obj_min';
end
num_ = length(OBJECTIVES);
newcolors=turbo(num_);
all_marks = {'h','o','+','*','<','x','p','d','^','v','>','.','s'};

figure()
if strcmp(format, 'obj_min')
    obj_best = Inf;
    for k = 1:num_
        obj_best = min(obj_best, min( OBJECTIVES{k}) );
    end
    obj_best = obj_best*ones(num_,1)-1e-10;
else
    obj_best= ones(num_,1);
    for k = 1:num_
        obj_best(k) = min( OBJECTIVES{k});
    end
    
end
mh_ = [];
for k = 1:num_
    
    if length(TIMES{k})==1
        tGrid = linspace(0,TIMES{k},length(OBJECTIVES{k}));
    else
        tGrid = TIMES{k};
    end
    if strcmp(NAMES{k},name)
        h=semilogy( tGrid, cummin( OBJECTIVES{k} - obj_best(k)), 'b-.','DisplayName', NAMES{k}  );
        set(h,'linewidth',4);
        hold on
    else
        obj_ = cummin(  OBJECTIVES{k} - obj_best(k));
        sp_ = ceil(length(obj_)/10);
        h = semilogy( tGrid(1:sp_:end), obj_(1:sp_:end),'Color', newcolors(k,:),'LineStyle', 'none' , 'Marker',  all_marks{mod(k,13)+1}, 'Markersize',10  );
        hold on
        h1=semilogy( tGrid, obj_, 'Color', newcolors(k,:),'DisplayName', NAMES{k} );
        set(h,'linewidth',2);
        set(h1,'linewidth',1.5);
    end
    mh_ = [mh_, h];
    
end
legend(mh_,NAMES)
xlabel('time in seconds','fontsize',18);
ylabel('objective value error','fontsize',18);
set(gca,'fontsize',18)
end