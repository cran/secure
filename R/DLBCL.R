#' Chemotherapy data
#' 
#' Gene expression dataset from the patients with diffuse large-B-cell lymphoma (DLBCL) after chemotherapy
#' The data has been used for unsupervised analysis i.e. Biclustering. The data consists of expression levels of q = 661 genes from n =180 patients. Among the patients, 42, 51 and
#' 87 of them were classified to OxPhos, BCR and HR groups, respectively. The data thus
#' form an n by q matrix Y whose rows represent the subjects and columns correspond to the
#' genes
#'
#' @docType data
#' @usage data(DLBCL)
#' @format A data frame 
#' @keywords datasets
#' @examples
#' # data(DLBCL)
#' # p <- nrow(DLBCL); q <- ncol(DLBCL); n <- nrow(DLBCL)
#' # control <- secure.control(spU=0.95,spV=0.95)
#' # fit.DLBCL <- secure.path(Y = DLBCL,X = NULL,nrank=10,nlambda = 100,
#' #                         orthXU = TRUE, orthV = TRUE, control=control)
"DLBCL"
