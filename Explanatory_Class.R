books_2_film = read.csv("C:/MAMP/htdocs/msci_446_books/Book-to-Film_Complete.csv", header=T)
all_books = read.csv("C:/MAMP/htdocs/msci_446_books/Book_Dataset_Task2_Filtered.csv", header=T)
View(books_2_film)
View(all_books)

numrow1 = nrow(all_books) #number of all books
numcol1 = ncol(all_books)
numrow2 = nrow(books_2_film) #number of books in books_2_film
numcol2 = ncol(books_2_film) 

colnames(books_2_film)<-c(1:numcol2)

class_col = matrix(0,nrow = numrow1, ncol = 1)
class_col2= matrix(1,nrow = numrow2, ncol = 1)
exp_class = cbind(all_books,class_col)
book_film_class=cbind(books_2_film,class_col2)
colnames(exp_class)<-c(1:numcol1)
output = merge(exp_class,book_film_class,by.x=4, by.y=1)
View(output)
   #View(bookfilm)
   for(row2 in 1:numrow2){ #iterate through all books
     #bookall = all_books[row1,4]
    # toString(tolower(bookall))
     #for(row2 in 1: numrow2){ #iterate through books with film
       bookfilm = books_2_film[row2,1]
       toString(tolower(bookfilm))
   #View(exp_class[,4])
   if (!is.na(bookfilm)){
     
     if(bookfilm %in% exp_class[,4]){
       row_match = match(bookfilm, tolower(exp_class[,4]))
     #  if(identical(bookall,bookfilm)){
         exp_class[row_match,numcol1+1]= 1
      # }
   }
   #}
   
   }
 }
   
 View(exp_class)
 View(match(1, exp_class[,4]))