%% P2
% crea database:
% dalla directory/path personale andiamo a prendere il database Yale fornito
imagespath=dir('C:\Users\cecil\Documents\Universita\ElaborazioneImmagini\Progetto\Database_per_gli_elaborati_di_Tipo_2_20190202\Yale\Yale');
imagespath;
ImagesPath=imagespath(4:end,:);% Sono eliminati i primi tre elementi perche' non sono immagini

% creiamo una struct images in cui inserire solo le immagini(unit8) e le tipologie(char)/classi
images=struct;
for i=1:size(ImagesPath,1)
    images(i).images=imread([ImagesPath(i).folder,'/',ImagesPath(i).name]);
    images(i).tipologia = extractAfter(ImagesPath(i).name,".");
end

% Associamo un'etichetta numerica(double) ad ogni tipologia(char) 
for i=1:size(images,2)
    switch images(i).tipologia
        case 'glasses'
           images(i).etichetta = 1; 
        case 'happy'
            images(i).etichetta = 2;
        case 'leftlight'
            images(i).etichetta = 3;
        case 'noglasses'
            images(i).etichetta = 4;
        case 'normal'
            images(i).etichetta = 5;
        case 'rightlight'
            images(i).etichetta = 6;
        case 'sad'
            images(i).etichetta = 7;
        case 'sleepy'
            images(i).etichetta = 8;
        case 'surprised'
            images(i).etichetta = 9;
        case 'wink'
            images(i).etichetta = 10;
        case 'centerlight'
            images(i).etichetta = 11;
                  
    end
            
end

%% Creiamo le HOG feature di tutte le immagini

% Inseriamo le feature estratte nella struttura: database.condizione(a).feature(b).matrice.matrice 
% a= 1--> senza rumore e a=2,3,4,5 con rumore crescente addizionato all'immagine originale
% b= 1,2 e 3 sono le varie CellSize, rispettivamente [2,2] [4,4] [8,8]

% dobbiamo far variare con un ciclo "for" a da 1 a 5 per i vari livelli di rumore
% e un ciclo "for" b da 1 a 3 per le diverse CellSize.

c=[2 4 8]; %vettore che rappresenta le dimensioni CellSize di nostro interesse, varia con b
rumore = logspace(-5,-1,5); %funzione per definire il rumore
for a=1:5 % scorro rumori
   for b=1:3 % scorro le cell size
          for i=1:size(images,2) %164 immagini
           [database.condizione(a).feature(b).matrice(i).matrice,visualization] = extractHOGFeatures((im2double(images(i).images)+sqrt(rumore(a))*randn(size(images(i).images))), ...
               'CellSize',[c(b) c(b)]);
          end          
    
   end
end

%prova per visualizzare le HOG 
%imshow(database.dimensione.dominio(1).immagini(2).matrice(11).matrice);
%hold on
%plot(visualization)

%% Classificatore con logica leave one out
% da fare rumore 0.0010 per a = 3 con cellsize 2 e 4 (b = 1 e 2)(circa 12 ore)
v=1:size(images,2); %vettore degli indici delle immagini totali
for a =1:5
    for b=1:3 %cell size
        clear training_features;
        clear training_labels;
        clear test_feature;
        for i=1:size(images,2) % "i" indica la feature di test. (La feature testata è l'i-esima)
       
            t= setdiff(v,i); % vettore degli indici del training (tutte le features esclusa quella di test i-ma)
        
        % vettori contententi le features e le etichette per il training da passare al fitcecoc
            for k=1:size(t,2)%163 vettore senza indice di training
               training_features(k,:) = database.condizione(a).feature(b).matrice(t(k)).matrice;
               training_labels(k) =images(t(k)).etichetta;
            end
       
      
           %classifier fitcecoc in ingresso: feature e etichette per il training, logica "one versus all"
           classifier= fitcecoc(training_features, training_labels, 'Coding', 'onevsall');
           
            % vettore con la feature da testare con il classificatore
             test_feature = database.condizione(a).feature(b).matrice(i).matrice; 
             
              % stuct contenente la ground truth per confronto con predizione e per calcolare la matrice di confusione
             risultati.condizioni(a).cellsize(b).test(i).verita = images(i).etichetta; 
           
            % etichetta predetta dal classificatore per la feature testata.,
             risultati.condizioni(a).cellsize(b).test(i).predizioni= predict(classifier, test_feature);
          
        i % stampiamo l'indice per sapere a che ciclo è arrivato il programma
        end
    end
 end



%% Calcolo della matrice di confusione

for a=1:5 %rumori
    for b=1:3 % cell size
       
        % la funzione confusionmat accetta in ingresso due vettori:
        % creiamo due vettori con i dati contenuti nella struct "risultati" per verità e predizione
        for i=1:size(risultati.condizioni(a).cellsize(b).test,2) %fino a 164 
           verita(i,1) = risultati.condizioni(a).cellsize(b).test(i).verita;
           predizione(i,1) = risultati.condizioni(a).cellsize(b).test(i).predizioni;
        end
        % calcoliamo la matrice di confusione con "confusionmat"
        analisi_dati.condizioni(a).cellsize(b).matrix_confusione = confusionmat(verita,...
            predizione);
        
        % creiamo una struct con le analisi dei risultati: accuracy, recall, precision
        
        % calcolo accuracy
        analisi_dati.condizioni(a).cellsize(b).accuracy = trace(analisi_dati.condizioni(a).cellsize(b).matrix_confusione)/sum(sum(analisi_dati.condizioni(a).cellsize(b).matrix_confusione));
        
        d = diag(analisi_dati.condizioni(a).cellsize(b).matrix_confusione);
        for i=1:size(analisi_dati.condizioni(a).cellsize(b).matrix_confusione,1)
            %calcolo recall
            analisi_dati.condizioni(a).cellsize(b).recall(i).recall = d(i)/...
              sum(analisi_dati.condizioni(a).cellsize(b).matrix_confusione(i,:));
            %calcolo precision
            analisi_dati.condizioni(a).cellsize(b).precision(i).precision = d(i)/...
              sum(analisi_dati.condizioni(a).cellsize(b).matrix_confusione(:,i));
        end   
    end
end

% prendo i counter
for i= 1: 5
    for j =1 :3
    counter(i,j) = trace(analisi_dati.condizioni(i).cellsize(j).matrix_confusione);
    end
end

for i = 1 :3 
    media_counter(i)=mean(counter(:,i));
    dev_counter(i) = std(counter(:,i));
end


% prendo le accuratezze
for i= 1: 5
    for j =1 :3
    accuratezza(i,j) = (counter(i,j)/164)*100;
    end
end


for i = 1 :3 
    media_accuracy(i)=mean(accuratezza(:,i));
    dev_accuracy(i) = std(accuratezza(:,i));
end


%cell size 1
 for i = 1 :11
       for a = 1 :5
           %calcolo racall        
           recall(a,i) =  analisi_dati.condizioni(a).cellsize(1).recall(i).recall;
           precision(a,i) = analisi_dati.condizioni(a).cellsize(1).precision(i).precision;
       end 
           media_recall_1(i) = mean(recall(:,i));
           media_precision_1(i) = mean(precision(:,i));
           dev_recall_1(i) = std(recall(:,i));
           dev_precision_1(i) = std(precision(:,i));
 end

 
 %cell size 2
 for i = 1 :11
       for a = 1 :5
           %calcolo racall        
           recall(a,i) =  analisi_dati.condizioni(a).cellsize(2).recall(i).recall;
           precision(a,i) = analisi_dati.condizioni(a).cellsize(2).precision(i).precision;
       end 
           media_recall_2(i) = mean(recall(:,i));
           media_precision_2(i) = mean(precision(:,i));
           dev_recall_2(i) = std(recall(:,i));
           dev_precision_2(i) = std(precision(:,i));
 end
 
  %cell size 3
 for i = 1 :11
       for a = 1 :5
           %calcolo racall        
           recall(a,i) =  analisi_dati.condizioni(a).cellsize(3).recall(i).recall;
           precision(a,i) = analisi_dati.condizioni(a).cellsize(3).precision(i).precision;
       end 
           media_recall_3(i) = mean(recall(:,i));
           media_precision_3(i) = mean(precision(:,i));
           dev_recall_3(i) = std(recall(:,i));
           dev_precision_3(i) = std(precision(:,i));
 end
 
 


%% grafici
% counter
x=1:3;
y=media_counter;
err = dev_counter;
scatter(x,y,'o','b','LineWidth', 2)
hold on
errorbar(x,y,err,'.r')
grid on
xlim([0 4])
ylim([0 164])
title('Media immagini identificate correttamente')
ylabel('N� di etichette stimate correttamente')
legend({'Media','deviazione standard'})
xticks(0:4)
yticks(0:10:164)
xticklabels({' ','cell size 2x2','cell size 4x4','cell size 8x8'})
xtickangle(45)

figure
%accuracy
x=1:3 ;
y=media_accuracy;
err1 = dev_accuracy;
scatter(x,y,'o','b','LineWidth', 2)
hold on
errorbar(x,y,err1,'.r')
grid on
xlim([0 4])
ylim([0 100])
title('Accuratezza per le diverse cell size')
ylabel('Accuratezza [%]')
legend({'Media','deviazione standard'})
xticks(0:4)
yticks(0:10:100)
xticklabels({' ','cell size 2x2','cell size 4x4','cell size 8x8'})
xtickangle(45)


figure
%recall per tutte e 11
x=1:1:11;
scatter(x,media_recall_1,'o','b','LineWidth', 2)
hold on
errorbar(x,media_recall_1,dev_recall_1,'.r')
grid on
legend({'Media','deviazione standard'})
xlim([0 12])
title({'Recall cell size 2x2'})
ylabel('Recall')
xticks(0:1:12)
xticklabels({' ','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy',...
    'surprised','wink','centerlight',' '})
xtickangle(45)

figure
%precision per tutte e 11
x=1:1:11;
scatter(x,media_precision_1,'o','b','LineWidth', 2)
hold on
errorbar(x,media_precision_1,dev_precision_1,'.r')
grid on
legend({'Media','deviazione standard'})
xlim([0 12])
title({'Precision cell size 2x2'})
ylabel('Precision')
xticks(0:1:12)
xticklabels({' ','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy',...
    'surprised','wink','centerlight',' '})
xtickangle(45)


figure
%% cell size 4x4
%recall per tutte e 11
x=1:1:11;
scatter(x,media_recall_2,'o','b','LineWidth', 2)
hold on
errorbar(x,media_recall_2,dev_recall_2,'.r')
grid on
legend({'Media','deviazione standard'})
xlim([0 12])
title({'Recall cell size 4x4'})
ylabel('Recall')
xticks(0:1:12)
xticklabels({' ','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy',...
    'surprised','wink','centerlight',' '})
xtickangle(45)

figure
%precision per tutte e 11
x=1:1:11;
scatter(x,media_precision_2,'o','b','LineWidth', 2)
hold on
errorbar(x,media_precision_2,dev_precision_2,'.r')
grid on
legend({'Media','deviazione standard'})
xlim([0 12])
title({'Precision cell size 4x4'})
ylabel('Precision')
xticks(0:1:12)
xticklabels({' ','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy',...
    'surprised','wink','centerlight',' '})
xtickangle(45)


figure
%% cell size 8x8
%recall per tutte e 11
x=1:1:11;
scatter(x,media_recall_3,'o','b','LineWidth', 2)
hold on
errorbar(x,media_recall_3,dev_recall_3,'.r')
grid on
legend({'Media','deviazione standard'})
xlim([0 12])
title({'Recall cell size 8x8'})
ylabel('Recall')
xticks(0:1:12)
xticklabels({' ','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy',...
    'surprised','wink','centerlight',' '})
xtickangle(45)

figure
%precision per tutte e 11
x=1:1:11;
scatter(x,media_precision_3,'o','b','LineWidth', 2)
hold on
errorbar(x,media_precision_3,dev_precision_3,'.r')
grid on
legend({'Media','deviazione standard'})
xlim([0 12])
title({'Precision cell size 8x8'})
ylabel('Precision')
xticks(0:1:12)
xticklabels({' ','glasses','happy','leftlight','noglasses','normal','rightlight','sad','sleepy',...
    'surprised','wink','centerlight',' '})
xtickangle(45)

