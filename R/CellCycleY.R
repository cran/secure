#' Cell cycle data
#'
#' The Eukariotic cell cycle data were generated using alpha factor arrest method,
#' consisting of RNA levels measured every 7 minutes for 119 minutes with a total 
#' of 18 time points covering two cell cycle of 1790 genes.
#'
#'
#' @docType data
#'
#' @usage data(CellCycleY)
#'
#' @format A data frame
#' @keywords datasets
#' @examples
#' # data(CellCycleY) # Y
#' # data(CellCycleX) # X
#' # n <- nrow(CellCycleY); p <- ncol(CellCycleX); q <-  ncol(CellCycleY)
#' # control <- secure.control(spU=160/p,spV=1)
#' # fit.cycle <- secure.path(CellCycleY, CellCycleX, nrank = 10, nlambda = 100,
#' #                   control = control)
"CellCycleY"