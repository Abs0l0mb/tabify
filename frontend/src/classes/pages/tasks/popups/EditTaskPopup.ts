'use strict';

import {
    Popup,
    Api,
    Form,
    FormField,
    TextInput,
    BigTextInput,
    NumberInput,
    SelectInput
} from '@src/classes';

export class EditTaskPopup extends Popup {

    private form: Form;

    private title: FormField;
    private description: FormField;
    private category: FormField;
    private estimatedHours: FormField;

    constructor(private taskId: number, private teamId: number) {

        super({
            validText: 'Update',
            cancellable: true,
            title: `Edit task ${taskId}`,
        });

        this.build();
    }

    /*
    **
    **
    */
    private async build() : Promise<void> {

        this.form = new Form(this.content);

        //=====
        //TITLE
        //=====

        this.title = this.form.add(new TextInput({
            label: 'Title',
            mandatory: true
        }));

        this.title.linkToErrorKey('title');

        //===========
        //DESCRIPTION
        //===========

        this.description = this.form.add(new BigTextInput({
            label: 'Description',
            mandatory: true
        }));

        this.description.linkToErrorKey('description');

        //========
        //CATEGORY
        //========
        
        this.category = this.form.add(new SelectInput({
            label: 'Category',
            items: await this.getTeamCategories(),
            mandatory: true
        }));

        this.category.linkToErrorKey('categoryId');

        //===============
        //ESTIMATED HOURS
        //===============

        this.estimatedHours = this.form.add(new NumberInput({
            label: 'Estimated hours',
            mandatory: true
        }));

        this.estimatedHours.linkToErrorKey('estimatedHours');

        this.populate();
    }

    /*
    **
    **
    */
    private async populate() : Promise<void> {
        
        try {

            let data: any = await Api.get('/task', {
                id: this.taskId
            });

            this.title.input.setValue(data.title);
            this.description.input.setValue(data.description);
            this.category.input.setValue(data.category_id);
            this.estimatedHours.input.setValue(data.estimated_hours);

            this.ready();
            
        } catch(error: any) {

            console.log(error);
        }
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/task/update', {
                id: this.taskId,
                title: this.title.input.getValue(),
                description: this.description.input.getValue(),
                categoryId: this.category.input.getValue(),
                estimatedHours: this.estimatedHours.input.getValue()
            });

            this.hide();
                        
        } catch(error: any) {

            this.form.displayError(error);
            this.validButton.unload();
        }
    }

    /*
    **
    **
    */
    private async getTeamCategories() : Promise<any[]> {
        
        console.log(this);

        let output : any[] = [];

        let data = await Api.get('/team/categories', {
            teamId: this.teamId
        });

        for (let row of data) {
            output.push({
                label: row.title,
                value: row.id
            });
        }

        return output;
    }
}