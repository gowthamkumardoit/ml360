import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  MatToolbarModule, MatButtonModule, MatCardModule, MatInputModule,
  MatFormFieldModule, MatListModule, MatIconModule, MatSnackBarModule, MatSliderModule,
  MatSlideToggleModule,
  MatSelectModule,
  MatStepperModule,
  MatTooltipModule,
  MatAutocompleteModule,
  MatBadgeModule,
  MatCheckboxModule,
  MatBottomSheetModule,
  MatProgressSpinnerModule
} from '@angular/material';

@NgModule({
  declarations: [],
  imports: [
    CommonModule,
    MatToolbarModule,
    MatButtonModule,
    MatCardModule,
    MatInputModule,
    MatFormFieldModule,
    MatListModule,
    MatIconModule,
    MatSliderModule,
    MatSlideToggleModule,
    MatSelectModule,
    MatStepperModule,
    MatTooltipModule,
    MatAutocompleteModule,
    MatBadgeModule,
    MatCheckboxModule,
    MatSnackBarModule,
    MatBottomSheetModule,
    MatProgressSpinnerModule
  ],
  exports: [
    MatToolbarModule,
    MatButtonModule,
    MatCardModule,
    MatInputModule,
    MatFormFieldModule,
    MatListModule,
    MatIconModule,
    MatSliderModule,
    MatSlideToggleModule,
    MatSelectModule,
    MatStepperModule,
    MatTooltipModule,
    MatAutocompleteModule,
    MatBadgeModule,
    MatCheckboxModule,
    MatSnackBarModule,
    MatBottomSheetModule,
    MatProgressSpinnerModule
  ]
})
export class MaterialModule { }
