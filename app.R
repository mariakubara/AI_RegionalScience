##############################################################
# Combined Shiny App with Two Tabs: 
#   1) Interactive Map Tab
#   2) Slider-Based Prediction Tool Tab
##############################################################

# -- Load required libraries --
library(shiny)
library(shinythemes)
library(leaflet)
library(sf)
library(dplyr)
library(xgboost)
library(shinycssloaders)  # for the withSpinner() functionality

# # -- Load any data or models you need globally --
# # The final XGBoost model:
# xgb_final_model <- readRDS("final_models/xgb_final_model_labels.rds")
# 
# load(file="dataForShinyApp.RData")
# # data and WOJ.maz
# 
# # Convert data to sf if needed
# data.sf <- st_as_sf(data, coords = c("lon","lat"), crs = "+proj=longlat +datum=NAD83")
# 
# sample <- sample(nrow(data), size = 50000, replace = FALSE)
# 
# data.sf <- data.sf[sample, ]
# data <- data[sample,]
# 
# # Ensure spatial data is properly projected if needed:
# if (st_crs(WOJ.maz)$epsg != 4326) {
#   WOJ.maz <- st_transform(WOJ.maz, crs = 4326)
# }
# 
# data.sf <- st_transform(data.sf, crs = 4326)
# 

# dataset includes a sample of 50 000 firms (needing to downsize due to camputation issues)
# save(data, data.sf, WOJ.maz, xgb_final_model, file="dataForShinyAppSample.RData")

load(file="dataForShinyAppSample.RData")


##############################################################
# Define UI
##############################################################
ui <- navbarPage(
  title = "Business sector prediction app",
  theme = shinytheme("flatly"),
  
  # ---- 1) FIRST TAB: MAP ----
  tabPanel(
    title = "Interactive Map",
    sidebarLayout(
      sidebarPanel(
        h4("Instructions"),
        helpText("Click on the map to select a location. The app will find the nearest point 
                 with available data and display the predicted probabilities for each 
                 business sector at that location."),
        br(),
        actionButton("reset_map", "Reset Map")
      ),
      mainPanel(
        leafletOutput("map", height = 600),
        br(),
        h3("Map Prediction Results"),
        # Wrap the table in a spinner
        withSpinner(
          tableOutput("prediction_map_output"), 
          type = 6  # choose your favorite spinner type
        )
      )
    )
  ),
  
  # ---- 2) SECOND TAB: SLIDER-BASED PREDICTION ----
  tabPanel(
    title = "Slider Prediction Tool",
    
    sidebarLayout(
      sidebarPanel(
        h3("Input Features"),
        helpText("Adjust the sliders and checkboxes for each feature, then press 'Predict'."),
        actionButton("reset_slider", "Reset Inputs"),
        br(),
        br(),
        uiOutput("feature_inputs"),
        actionButton("predict_btn", "Predict", class = "btn-primary")
      ),
      mainPanel(
        h3("Slider Prediction Results"),
        # Wrap the table in a spinner
        withSpinner(
          tableOutput("prediction_slider_output"), 
          type = 6
        )
      )
    )
  )
)

##############################################################
# Define Server Logic
##############################################################
server <- function(input, output, session) {
  
  #################################################################
  #  A) MAP TAB LOGIC
  #################################################################
  
  # Create a reactiveValues to store clicked location & other map info
  rvMap <- reactiveValues(clicked_location = NULL)
  
  # Render the leaflet map
  output$map <- renderLeaflet({
    leaflet() %>%
      addProviderTiles(providers$CartoDB.Positron) %>%
      addPolygons(data = WOJ.maz, color = "#444444", weight = 1, smoothFactor = 0.5,
                  opacity = 1.0, fillOpacity = 0.2,
                  fillColor = "lightblue",
                  highlightOptions = highlightOptions(color = "white", weight = 2,
                                                      bringToFront = TRUE)) %>%
      setView(
        lng = mean(st_coordinates(WOJ.maz)[,1]), 
        lat = mean(st_coordinates(WOJ.maz)[,2]), 
        zoom = 8
      )
  })
  
  # Observe the map click event
  observeEvent(input$map_click, {
    click <- input$map_click
    rvMap$clicked_location <- c(click$lng, click$lat)
    
    # Create an sf point from the clicked location
    clicked_point_sf <- st_sfc(st_point(rvMap$clicked_location), crs = 4326)
    
    # Find the nearest point in data.sf
    nearest_index <- st_nearest_feature(clicked_point_sf, data.sf)
    nearest_point <- data.sf[nearest_index, ]
    
    # Extract the features of the nearest point
    features <- nearest_point %>% 
      st_set_geometry(NULL) %>% 
      select(xgb_final_model$feature_names)
    
    # Check if all needed features are present
    missing_features <- setdiff(xgb_final_model$feature_names, names(features))
    if (length(missing_features) > 0) {
      stop("Required features missing in the nearest point data: ", 
           paste(missing_features, collapse = ", "))
    }
    
    # Convert features to matrix
    features_matrix <- as.matrix(features)
    
    # Make prediction
    prediction_prob <- predict(xgb_final_model, features_matrix)
    
    # Suppose we have a factor in data$class4 for the target class
    class_labels <- levels(data$class4)
    
    # Reshape probabilities if multiple classes
    num_classes <- length(class_labels)
    prediction_prob <- matrix(prediction_prob, ncol = num_classes, byrow = TRUE)
    colnames(prediction_prob) <- class_labels
    
    # Convert to data frame for display
    prediction_df <- data.frame(
      Class = class_labels,
      Probability = as.numeric(prediction_prob)
    ) %>%
      mutate(Probability = round(Probability * 100, 2)) %>%
      arrange(desc(Probability))
    
    output$prediction_map_output <- renderTable({
      prediction_df
    }, digits = 2)
    
    # Update the map with markers
    leafletProxy("map") %>%
      clearMarkers() %>%
      addMarkers(lng = click$lng, lat = click$lat, 
                 popup = "Clicked Location", label = "You clicked here") %>%
      addMarkers(data = nearest_point, 
                 popup = "Nearest Data Point", label = "Nearest data point")
  })
  
  # Reset map
  observeEvent(input$reset_map, {
    rvMap$clicked_location <- NULL
    output$prediction_map_output <- renderTable(NULL)
    leafletProxy("map") %>% clearMarkers()
  })
  
  #################################################################
  #  B) SLIDER TAB LOGIC
  #################################################################
  
  # Identify feature columns from the model
  feature_cols <- xgb_final_model$feature_names
  # A typical factor with class labels (if needed)
  class_labels <- levels(data$class4)
  
  # Create the dynamic UI for features
  output$feature_inputs <- renderUI({
    inputs <- lapply(feature_cols, function(feature) {
      # If ends with ".s", treat as continuous (slider)
      if (grepl("\\.s$", feature)) {
        sliderInput(
          inputId = feature,
          label = feature,
          min = 0,
          max = 1,
          value = 0.5,
          step = 0.01
        )
      } else {
        # else treat as dummy
        checkboxInput(
          inputId = feature,
          label = feature,
          value = FALSE
        )
      }
    })
    do.call(tagList, inputs)
  })
  
  # Observe reset button
  observeEvent(input$reset_slider, {
    lapply(feature_cols, function(feature) {
      if (grepl("\\.s$", feature)) {
        updateSliderInput(session, feature, value = 0.5)
      } else {
        updateCheckboxInput(session, feature, value = FALSE)
      }
    })
  })
  
  # Predict when "Predict" is clicked
  observeEvent(input$predict_btn, {
    input_values <- sapply(feature_cols, function(feature) {
      value <- input[[feature]]
      if (is.logical(value)) {
        as.numeric(value)  # Convert TRUE/FALSE -> 1/0
      } else {
        value
      }
    })
    input_df <- as.data.frame(t(input_values))
    colnames(input_df) <- feature_cols
    
    # Convert to matrix
    input_matrix <- as.matrix(input_df)
    
    # Make prediction
    prediction_prob <- predict(xgb_final_model, input_matrix)
    
    # If multiclass, name them
    prediction_prob_named <- setNames(as.numeric(prediction_prob), class_labels)
    
    prediction_df <- data.frame(
      Class = class_labels,
      Probability = prediction_prob_named
    )
    
    output$prediction_slider_output <- renderTable({
      prediction_df
    }, digits = 4)
  })
  
}  # end server

##############################################################
# Run the application
##############################################################
shinyApp(ui = ui, server = server)
