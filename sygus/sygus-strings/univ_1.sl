(set-logic SLIA)
 
(synth-fun f ((col1 String) (col2 String)) String
    ((Start String (ntString))
     (ntString String (col1 col2 " " ","
                       (str.++ ntString ntString)
                       (str.replace ntString ntString ntString)
                       (str.at ntString ntInt)
                       (int.to.str ntInt)
                       (str.substr ntString ntInt ntInt)))
      (ntInt Int (0 1 2
                  (+ ntInt ntInt)
                  (- ntInt ntInt)
                  (str.len ntString)
                  (str.to.int ntString)
                  (str.indexof ntString ntString ntInt)))
      (ntBool Bool (true false
                    (str.prefixof ntString ntString)
                    (str.suffixof ntString ntString)
                    (str.contains ntString ntString)))))


(declare-var col1 String)
(declare-var col2 String)

(constraint (= (f "University of Pennsylvania" "Phialdelphia, PA, USA") 
                  "University of Pennsylvania, Phialdelphia, PA, USA"))

(constraint (= (f "Cornell University" "Ithaca, New York, USA") 
                  "Cornell University, Ithaca, New York, USA"))

(constraint (= (f "University of Maryland College Park" "College Park, MD")   
                  "University of Maryland College Park, College Park, MD"))

(constraint (= (f "University of Michigan" "Ann Arbor, MI, USA") 
                  "University of Michigan, Ann Arbor, MI, USA"))

(constraint (= (f "Yale University" "New Haven, CT, USA") 
                  "Yale University, New Haven, CT, USA"))

(constraint (= (f "Columbia University" "New York, NY, USA") 
                  "Columbia University, New York, NY, USA"))
 
(check-synth)
