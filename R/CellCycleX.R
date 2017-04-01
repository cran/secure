#' Cell cycle data
#' 
#' The chromatin immunoprecipitation (ChIP) data (X) contain complete binding 
#' information of a subset of 1790 genes for a total of 113 transcription 
#' factors.
#'
#' @docType data
#'
#' @usage data(CellCycleX)
#'
#' @format A data frame
#'
#' @keywords datasets

#' @examples
#' # data(CellCycleY) # Y
#' # data(CellCycleX) # X
#' # n <- nrow(CellCycleY); p <- ncol(CellCycleX); q <-  ncol(CellCycleY)
#' # control <- secure.control(spU=160/p,spV=1)
#' # fit.cycle <- secure.path(CellCycleY, CellCycleX, nrank = 10, nlambda = 100,
#' #                   control = control)
"CellCycleX"