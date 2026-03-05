'use strict';

import {
    Popup,
    Api,
    Form,
    FormField,
    TextInput,
    BigTextInput,
    NumberInput,
    SelectInput,
    InlineInputsContainer,
    DateInput,
    TaskData,
    TaskStatus
} from '@src/classes';

export class AddTaskPopup extends Popup {

    private form: Form;

    private title: FormField;
    private description: FormField;
    private category: FormField;
    private estimatedHours: FormField;
    private todoDate: FormField;
    private doneDate: FormField;

    constructor(private teamId: number, private status: TaskStatus, private taskData?: TaskData) {

        super({
            validText: 'Add',
            cancellable: true,
            title: `Add ${status.toLowerCase()} task`,
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

        if (this.taskData) {
            this.title.input.setValue(this.taskData.title);
            this.description.input.setValue(this.taskData.description);
            this.category.input.setValue(this.taskData.category_id);
            this.estimatedHours.input.setValue(this.taskData.estimated_hours);
        }

        if (this.status === 'DONE') {

            //==========
            //DONE DATES
            //==========

            const inline = new InlineInputsContainer(this.form);

            this.todoDate = this.form.add(new DateInput({
                label: 'Start date'
            }), inline);

            this.todoDate.linkToErrorKey('start');

            this.doneDate = this.form.add(new DateInput({
                label: 'End date'
            }), inline);

            this.doneDate.linkToErrorKey('end');
        }

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            const data: any = {
                teamId: this.teamId,
                title: this.title.input.getValue(),
                description: this.description.input.getValue(),
                categoryId: this.category.input.getValue(),
                estimatedHours: this.estimatedHours.input.getValue(),
                status: this.status
            };

            if (this.status === 'DONE') {
                data.todoDate = this.todoDate.input.getValue();
                data.doneDate = this.doneDate.input.getValue();
            }

            const id = await Api.post('/team/tasks/add', data);

            this.emit('success', id);
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