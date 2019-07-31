import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatToolbarModule, MatButtonModule, MatCardModule, MatInputModule,
          MatFormFieldModule, MatListModule, MatIconModule, MatSnackBarModule, MatSliderModule,
          MatSlideToggleModule,
          MatSelectModule,
          MatStepperModule,
          MatTooltipModule,
          MatAutocompleteModule,
          MatBadgeModule,
          MatCheckboxModule,
          MatBottomSheetModule} from '@angular/material';

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
    MatBottomSheetModule
  ],
  exports : [
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
    MatBottomSheetModule
  ]
})
export class MaterialModule { }
